/** Save and retrieve session voice clips through IndexedDB. */
const AUDIO_DATABASE = "mfq.studio.audio.v1";
const AUDIO_STORE = "clips";

/** Open the audio cache database, creating a clip-ID-indexed object store on first use. */
function audioDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(AUDIO_DATABASE, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(AUDIO_STORE)) {
        request.result.createObjectStore(AUDIO_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/** Save a voice clip to IndexedDB and return only after the transaction completes. */
export async function saveVoiceClip(id: string, blob: Blob): Promise<void> {
  const database = await audioDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(AUDIO_STORE, "readwrite");
    transaction.objectStore(AUDIO_STORE).put(blob, id);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

/** Read cached audio by clip ID, returning null when it does not exist. */
export async function loadVoiceClip(id: string): Promise<Blob | null> {
  const database = await audioDatabase();
  return new Promise((resolve, reject) => {
    const request = database.transaction(AUDIO_STORE).objectStore(AUDIO_STORE).get(id);
    request.onsuccess = () => resolve((request.result as Blob | undefined) ?? null);
    request.onerror = () => reject(request.error);
  });
}
