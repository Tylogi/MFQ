/** Retain transitional source contracts for the real-time voice state machine, turn ownership, and defaults; this is not a behavior test, and audioCodec/AudioDevices behavior coverage does not replace it. */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const sourceRoot = resolve(process.cwd(), 'src');

/** Read product implementation only, excluding tests and type declarations so tests cannot satisfy their own contracts. */
function readSources(...paths: string[]): string {
  function collect(location: string): string[] {
    if (statSync(location).isDirectory()) {
      return readdirSync(location).sort().flatMap((name) => collect(join(location, name)));
    }
    return /\.tsx?$/.test(location) && !/\.(test|spec|d)\.tsx?$/.test(location)
      ? [readFileSync(location, 'utf8').replace(/\r\n/g, '\n')]
      : [];
  }
  return paths.flatMap((path) => collect(join(sourceRoot, path))).join('\n');
}

const REALTIME_AUDIO = readSources('realtimeAudio.ts', 'features/voice');
const APP = readSources('features/chat/ChatProvider.tsx', 'features/chat/hooks', 'features/settings', 'features/voice/useVoiceConversation.ts');

describe('describes voiceContracts test behavior 1', () => {
  it('test_voice_component_prompt_requires_an_explicit_full_duplex_selection', () => {
    const chat = readSources('features/chat/components/ChatInputArea.tsx');
    const toolbar = readSources('features/chat/components/ChatToolbar.tsx');
    expect(chat).toContain('active?.mode === \'full_duplex\'');
    expect(chat).toContain('model_capabilities.features.audio_output');
    expect(chat).toContain('!inference.realtimeAvailable');
    expect(chat).toContain('needsVoiceOutputComponent && voiceComponent');
    expect(toolbar).toContain('features.audio_input');
    expect(toolbar).toContain('features.full_duplex');
  });

  it('test_studio_drains_duplex_output_after_microphone_capture_stops', () => {
    expect(REALTIME_AUDIO).toContain('MAX_RESPONSE_DRAIN_STEPS');
    expect(REALTIME_AUDIO).toContain('event.end_of_turn === true');
    expect(REALTIME_AUDIO).toContain('this.sendInput(new Float32Array(CHUNK_SAMPLES))');
    expect(REALTIME_AUDIO).toContain('this.callbacks.onText(target.buffer.sessionId, target.buffer.text)');
  });

  it('test_studio_preserves_resampling_phase_across_audio_worklet_blocks', () => {
    // Chunk phase is covered by multi-rate chunked/whole-buffer equivalence tests in features/voice/audioCodec.test.ts.
    // Keep only device sample-rate wiring here; this neither checks resampler internals nor verifies voice-turn ownership.
    const devices = readSources('features/voice/AudioDevices.ts');
    expect(readSources('features/voice/audioProtocol.ts')).toContain('const INPUT_RATE = 16_000;');
    expect(devices).toContain("import { INPUT_RATE, OUTPUT_RATE } from './audioProtocol';");
    expect(devices).toContain('new AudioContext({ sampleRate: INPUT_RATE })');
  });

  it('test_studio_uses_the_model_bound_duplex_system_prompt', () => {
    expect(APP).toContain('modeTemplateSettings');
    expect(APP).toContain('runtime?.duplex_sampling_defaults ?? realtime?.defaults');
    expect(REALTIME_AUDIO).toContain('system_prompt: config.systemPrompt');
    expect(REALTIME_AUDIO).toContain('text_repetition_penalty: config.repetitionPenalty');
    expect(APP).toContain('submitText(text, realtimeSessionConfig(active.id))');
  });

  it('test_realtime_turns_remain_bound_to_the_session_that_created_them', () => {
    expect(REALTIME_AUDIO).toContain('sessionId: string;');
    expect(REALTIME_AUDIO).toContain('private clientSessionId: string | null = null');
    expect(REALTIME_AUDIO).toContain('sessionId: buffer.sessionId');
    expect(APP).toContain('onTurn: ({ id, sessionId, text, audio })');
    const voice = readSources('features/voice/useVoiceConversation.ts');
    const start = voice.indexOf('onInputStart:');
    const end = voice.indexOf('// The controller reads');
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    const voice_callbacks = voice.slice(start, end);
    expect(voice_callbacks).not.toContain('activeIdRef');
  });

  it('test_full_duplex_user_speech_visually_splits_assistant_turns', () => {
    expect(REALTIME_AUDIO).toContain('SPEECH_RMS_THRESHOLD');
    expect(REALTIME_AUDIO).toContain('this.finishTurn();\n    this.inputTurnId = crypto.randomUUID()');
    expect(REALTIME_AUDIO).toContain('this.callbacks.onInputStart');
    expect(REALTIME_AUDIO).toContain('this.callbacks.onInputEnd');
    expect(readSources('features/voice/useVoiceConversation.ts')).toContain('role: \'user\'');
    expect(APP).toContain('message.pending');
  });

  it('test_full_duplex_routes_pre_interrupt_response_tails_back_to_the_old_turn', () => {
    expect(REALTIME_AUDIO).toContain('private pendingResponseTurns');
    expect(REALTIME_AUDIO).toContain('private responseTurnIds');
    expect(REALTIME_AUDIO).toContain('private currentInputTurnId');
    expect(REALTIME_AUDIO).toContain('private responseMessageIds');
    expect(REALTIME_AUDIO).toContain('private completedTurns');
    expect(REALTIME_AUDIO).toContain('this.lastCompletedByInputTurn.get(inputTurnId)');
    expect(REALTIME_AUDIO).toContain('this.publishTurn(target.buffer)');
    expect(APP).toContain('message.id === id ? { ...message, text }');
  });

  it('test_closing_a_full_duplex_microphone_stops_instead_of_forcing_speech', () => {
    expect(REALTIME_AUDIO).not.toContain('finishFullDuplexInput');
    expect(REALTIME_AUDIO).toContain('} else if (this.audio.capturing) {\n      await this.stop();');
    expect(REALTIME_AUDIO).toContain('this.stopPlayback();');
  });

  it('verifies voiceContracts test behavior 2', () => {
    const controller = readSources('features/voice/RealtimeAudioController.ts');
    for (const marker of ['heldHalfDuplexChunk', 'forceListen: true', 'forceSpeak: true', 'event.type === "response.step.done"']) {
      expect(controller).toContain(marker);
    }
  });

  it('verifies voiceContracts test behavior 3', () => {
    expect(readSources('features/voice/RealtimeAudioController.ts')).toContain('const SPEAK_TOKENS = 20;');
    expect(readSources('features/voice/AudioDevices.ts')).toContain('const PLAYBACK_DELAY_SECONDS = 0.2;');
    const provider = readSources('features/chat/ChatProvider.tsx');
    expect(provider).not.toContain('REALTIME_SYSTEM_PROMPTS');
    expect(provider).toContain('systemPrompt: value.systemPrompt.trim()');
  });
});
