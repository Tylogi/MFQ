/** Manage one chat generation lifecycle, batched output, and history synchronization, isolating stale asynchronous writes. */
import { sessionsApi } from '../../../shared/api/resources/sessions';
import { ApiError } from '../../../shared/api/client';
import { streamResponse } from '../../../shared/api/responses';
import type { Message, ResponseResource, Session, StreamRequest } from '../../../shared/api/types';
import type { ResponseFrame } from '../../../shared/api/responseProtocol';

export type GenerationPhase =
  | 'idle'
  | 'submitting'
  | 'streaming'
  | 'stopping'
  | 'syncing'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface LiveOutput {
  reasoning: string;
  text: string;
  tools: string[];
}

export interface GenerationSnapshot {
  phase: GenerationPhase;
  sessionId: string | null;
  live: LiveOutput | null;
  error: string | null;
/** When history is unconfirmed, allow only resynchronization; never automatically resend the generation POST. */
  recoveryNeeded: boolean;
}

export interface ConversationSnapshot {
  session: Session;
  messages: Message[];
  responses: ResponseResource[];
}

interface GenerationServices {
  stream: (
    sessionId: string,
    request: StreamRequest,
    onFrame: (frame: ResponseFrame) => void,
    signal: AbortSignal,
    onAccepted?: () => void,
  ) => Promise<void>;
  cancel: (sessionId: string, signal: AbortSignal) => Promise<unknown>;
  synchronize: (sessionId: string, signal: AbortSignal) => Promise<ConversationSnapshot>;
}

export interface GenerationCallbacks {
/** Only the current request may publish a persisted snapshot; callers must still verify the selected session. */
  onSynchronized: (snapshot: ConversationSnapshot) => void;
/** Update the revision of the generating session without affecting other session selections. */
  onSessionState: (sessionId: string, state: Session['state'], revision: number) => void;
}

interface GenerationRun {
  id: string;
  sessionId: string;
  accepted: boolean;
  onAccepted?: () => void;
  controller: AbortController;
  synchronization: AbortController | null;
  cancellation: AbortController | null;
  cancelPromise: Promise<void> | null;
  cancelled: boolean;
  completed: boolean;
  error: string | null;
  output: LiveOutput;
}

const services: GenerationServices = {
  stream: streamResponse,
  cancel: (sessionId, signal) => sessionsApi.cancelResponse(sessionId, signal),
  synchronize: async (sessionId, signal) => {
    const [session, messages, responses] = await Promise.all([
      sessionsApi.getSession(sessionId, signal),
      sessionsApi.listMessages(sessionId, signal),
      sessionsApi.listResponses(sessionId, signal),
    ]);
    return { session, messages, responses };
  },
};

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
/** Determine whether the lifecycle still occupies the send channel, including stopping and history synchronization. */
export function isGenerationBusy(phase: GenerationPhase): boolean {
  return (
    phase === 'submitting' || phase === 'streaming' || phase === 'stopping' || phase === 'syncing'
  );
}
/** Create a separately subscribable generation controller so the UI observes only snapshots it needs. */
export class GenerationController {
  private snapshot: GenerationSnapshot = {
    phase: 'idle',
    sessionId: null,
    live: null,
    error: null,
    recoveryNeeded: false,
  };
  private listeners = new Set<() => void>();
  private active: GenerationRun | null = null;
  private flushTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private callbacks: GenerationCallbacks,
    private dependencies: GenerationServices = services,
  ) {}
/** Subscribe to snapshot changes; use the returned function to unsubscribe on React unmount. */
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };
/** Return the cached snapshot, preserving reference identity for the same revision for useSyncExternalStore. */
  getSnapshot = (): GenerationSnapshot => this.snapshot;
/** The app shell subscribes only to phases, so text deltas do not rerender the entire application. */
  getPhase = (): GenerationPhase => this.snapshot.phase;

  private publish(update: Partial<GenerationSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...update };
    this.listeners.forEach((listener) => listener());
  }

  private flush(run: GenerationRun): void {
    if (this.flushTimer !== null) clearTimeout(this.flushTimer);
    this.flushTimer = null;
    if (this.active !== run) return;
    this.publish({ live: { ...run.output, tools: [...run.output.tools] } });
  }

  private receive(run: GenerationRun, frame: ResponseFrame): void {
    if (this.active !== run || run.controller.signal.aborted || frame.session_id !== run.sessionId)
      return;
    const payload = frame.payload;
    if (payload.type === 'session.state') {
      this.callbacks.onSessionState(run.sessionId, payload.state, payload.revision);
      return;
    }
    if (payload.type === 'response.interrupted') run.cancelled = true;
    if (payload.type === 'response.completed') run.completed = true;
    if (payload.type === 'response.text.delta') run.output.text += payload.delta;
    else if (payload.type === 'response.reasoning.delta') run.output.reasoning += payload.delta;
    else if (payload.type === 'response.tool_call.delta') {
      run.output.tools[payload.index] =
        (run.output.tools[payload.index] ?? '') + payload.arguments_delta;
    } else return;
    if (this.snapshot.phase === 'submitting') this.publish({ phase: 'streaming' });
    if (this.flushTimer === null) this.flushTimer = setTimeout(() => this.flush(run), 32);
  }

  private accept(run: GenerationRun): void {
    if (this.active !== run || run.accepted) return;
    run.accepted = true;
    run.onAccepted?.();
  }
/** Start one generation; keep the send channel occupied until synchronization completes and never retry non-idempotent requests automatically. */
  async start(sessionId: string, request: StreamRequest, onAccepted?: () => void): Promise<void> {
    if (isGenerationBusy(this.snapshot.phase) || this.snapshot.recoveryNeeded) {
      throw new Error('Wait for the current response to synchronize before sending again');
    }
    const run: GenerationRun = {
      id: request.request_id,
      sessionId,
      accepted: false,
      onAccepted,
      controller: new AbortController(),
      synchronization: null,
      cancellation: null,
      cancelPromise: null,
      cancelled: false,
      completed: false,
      error: null,
      output: { text: '', reasoning: '', tools: [] },
    };
    this.active = run;
    this.publish({
      phase: 'submitting',
      sessionId,
      live: { ...run.output, tools: [] },
      error: null,
      recoveryNeeded: false,
    });
    try {
      await this.dependencies.stream(
        sessionId,
        request,
        (frame) => this.receive(run, frame),
        run.controller.signal,
        () => this.accept(run),
      );
    } catch (error) {
      if (!run.cancelled) run.error = describeError(error);
    } finally {
      if (this.active === run) {
        this.flush(run);
        if (run.cancelPromise) await run.cancelPromise;
        if (this.active === run) await this.synchronize(run);
      }
    }
  }
/** Stop the current generation; wait up to five seconds for cancellation, then abort reading and synchronize server state. */
  stop = async (): Promise<void> => {
    const run = this.active;
    if (
      !run ||
      run.cancelPromise ||
      (this.snapshot.phase !== 'submitting' && this.snapshot.phase !== 'streaming')
    )
      return;
    run.cancelled = true;
    this.publish({ phase: 'stopping' });
    run.cancelPromise = (async () => {
      const controller = new AbortController();
      run.cancellation = controller;
      const timer = setTimeout(
        () => controller.abort(new DOMException('Cancellation timed out', 'TimeoutError')),
        5000,
      );
      try {
        await this.dependencies.cancel(run.sessionId, controller.signal);
      } catch (error) {
        if (!(error instanceof ApiError && error.status === 409)) run.error = describeError(error);
      } finally {
        clearTimeout(timer);
        run.cancellation = null;
        run.controller.abort();
      }
    })();
    await run.cancelPromise;
  };

  private async synchronize(run: GenerationRun): Promise<void> {
    if (this.active !== run) return;
    this.publish({ phase: 'syncing' });
    const controller = new AbortController();
    run.synchronization = controller;
    const timer = setTimeout(
      () => controller.abort(new DOMException('History synchronization timed out', 'TimeoutError')),
      10000,
    );
    try {
      const persisted = await this.dependencies.synchronize(run.sessionId, controller.signal);
      if (this.active !== run) return;
      const response = persisted.responses.find((item) => item.request_id === run.id);
      if (run.completed && !response)
        throw new Error('The completed response is not yet available in history');
      if (response?.status === 'running' || persisted.session.state === 'processing') {
        throw new Error(
          'The server is still processing this response. Synchronize again before sending.',
        );
      }
      const storedOutput = Boolean(
        response?.output_message_id &&
        persisted.messages.some((item) => item.id === response.output_message_id),
      );
      if (response?.status === 'completed' && !storedOutput)
        throw new Error('The completed response is not yet available in history');
      if (response?.status === 'failed' && !run.error)
        run.error = 'The server could not complete this response';
      if (response) this.accept(run);
      this.callbacks.onSynchronized(persisted);
      const hasPartial = Boolean(
        run.output.text || run.output.reasoning || run.output.tools.length,
      );
      this.publish({
        phase:
          storedOutput && response?.status === 'completed'
            ? 'completed'
            : response?.status === 'cancelled' || run.cancelled
              ? 'cancelled'
              : run.error
                ? 'failed'
                : 'completed',
        live: storedOutput || !hasPartial ? null : this.snapshot.live,
        error: response?.status === 'completed' ? null : run.error,
        recoveryNeeded: false,
      });
    } catch (error) {
      if (this.active === run)
        this.publish({ phase: 'failed', error: describeError(error), recoveryNeeded: true });
    } finally {
      clearTimeout(timer);
      if (run.synchronization === controller) run.synchronization = null;
      controller.abort();
    }
  }
/** Reread only history and response status, retaining the original request ID and never restarting generation. */
  retrySynchronization = async (): Promise<void> => {
    if (this.active && this.snapshot.recoveryNeeded && !isGenerationBusy(this.snapshot.phase)) {
      await this.synchronize(this.active);
    }
  };
/** Invalidate callbacks and release readers, cancellation requests, and batch timers when the session changes or the component unmounts. */
  reset = (): void => {
    const previous = this.active;
    this.active = null;
    previous?.controller.abort();
    previous?.synchronization?.abort();
    previous?.cancellation?.abort();
    if (this.flushTimer !== null) clearTimeout(this.flushTimer);
    this.flushTimer = null;
    this.publish({
      phase: 'idle',
      sessionId: null,
      live: null,
      error: null,
      recoveryNeeded: false,
    });
  };
}
