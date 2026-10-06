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

/** Delete a cached clip after the IndexedDB transaction commits. */
export async function deleteVoiceClip(id: string): Promise<void> {
  const database = await audioDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(AUDIO_STORE, 'readwrite');
    transaction.objectStore(AUDIO_STORE).delete(id);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

/** Remove clips absent from retained voice history, including orphans left by earlier versions. */
export async function pruneVoiceClips(retainedIds: ReadonlySet<string>): Promise<void> {
  const database = await audioDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(AUDIO_STORE, 'readwrite');
    const store = transaction.objectStore(AUDIO_STORE);
    const request = store.getAllKeys();
    request.onsuccess = () => {
      for (const key of request.result) {
        if (!retainedIds.has(String(key))) store.delete(key);
      }
    };
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}
