
const API_BASE = "http://127.0.0.1:8000";


/* =========================================================
   TYPES
========================================================= */

export type VisionBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};


export type Detection = {
  class: string;
  confidence: number;
  box: VisionBox;
};


export type Face = {
  name: string;
  confidence: number;
  box: VisionBox;
};


export type VisionData = {
  detections: Detection[];
  faces: Face[];
};


export type ConversationMessage = {
  role: "user" | "assistant";
  content: string;
};


export type AssistantResponse = {
  success: boolean;
  route?: string;
  intent?: string;
  response?: string;
  error?: string;
};


export type TranscriptionResponse = {
  text?: string;
};


export type FileReadResponse = {
  success: boolean;
  filename?: string;
  extension?: string;
  characters?: number;
  text?: string;
  file_info?: {
    size_bytes?: number;
  };
  error?: string;
};


export type EnrollResponse = {
  success: boolean;
  message?: string;
  name?: string;
  total_people?: number;
  error?: string;
};


export type BackendStatus = {
  assistant?: string;
  status?: string;
  llm?: string;
  yolo?: string;
  face_recognition?: string;
  speech_recognition?: string;
  tts?: string;
  router?: string;
  weather?: string;
  calendar?: string;
  memory?: string;
  memory_count?: number;
  saved_faces?: number;
  file_reader?: string;
  supported_files?: string[];
  conversation_history?: string;
  context_aware_tools?: string;
  streaming?: string;
  command_actions?: string;
  max_history_messages?: number;
};


/* =========================================================
   ASSISTANT STREAMING
========================================================= */

export async function sendAssistantMessage(
  message: string,
  vision: VisionData = {
    detections: [],
    faces: [],
  },
  fileContext: string | null = null,
  history: ConversationMessage[] = [],
  signal?: AbortSignal,
  onChunk?: (chunk: string) => void
): Promise<AssistantResponse> {

  const recentHistory =
    history.slice(-12);


  const response =
    await fetch(
      `${API_BASE}/assistant`,
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body: JSON.stringify({
          message,

          vision,

          file_context:
            fileContext,

          history:
            recentHistory,
        }),

        signal,
      }
    );


  if (!response.ok) {

    throw new Error(
      `Server error: ${response.status}`
    );

  }


  const contentType =
    response.headers.get(
      "content-type"
    ) || "";


  /* =======================================================
     NORMAL JSON RESPONSE
  ======================================================= */

  if (
    !contentType.includes(
      "application/x-ndjson"
    )
  ) {

    const data =
      (await response.json()) as AssistantResponse;


    if (!data.success) {

      throw new Error(
        data.error ||
        "Assistant error"
      );

    }


    if (
      data.response &&
      onChunk
    ) {

      onChunk(
        data.response
      );

    }


    return data;

  }


  /* =======================================================
     STREAMING RESPONSE
  ======================================================= */

  if (!response.body) {

    throw new Error(
      "Streaming response body is unavailable."
    );

  }


  const reader =
    response.body.getReader();


  const decoder =
    new TextDecoder();


  let buffer = "";

  let fullResponse = "";

  let route: string | undefined;

  let intent: string | undefined;


  while (true) {

    const {
      value,
      done,
    } =
      await reader.read();


    if (done) {

      break;

    }


    buffer +=
      decoder.decode(
        value,
        {
          stream: true,
        }
      );


    const lines =
      buffer.split(
        "\n"
      );


    buffer =
      lines.pop() || "";


    for (
      const line of lines
    ) {

      const trimmed =
        line.trim();


      if (!trimmed) {

        continue;

      }


      let event: {
        type?: string;
        content?: string;
        message?: string;
        route?: string;
        intent?: string;
      };


      try {

        event =
          JSON.parse(
            trimmed
          );

      } catch {

        continue;

      }


      if (
        event.type ===
        "start"
      ) {

        route =
          event.route;

        intent =
          event.intent;

      }


      else if (
        event.type ===
        "chunk"
      ) {

        const chunk =
          event.content ||
          "";


        if (chunk) {

          fullResponse +=
            chunk;


          if (onChunk) {

            onChunk(
              chunk
            );

          }

        }

      }


      else if (
        event.type ===
        "error"
      ) {

        throw new Error(
          event.message ||
          "Assistant streaming error."
        );

      }

    }

  }


  if (
    buffer.trim()
  ) {

    try {

      const event =
        JSON.parse(
          buffer.trim()
        );


      if (
        event.type ===
        "chunk"
      ) {

        const chunk =
          event.content ||
          "";


        if (chunk) {

          fullResponse +=
            chunk;


          if (onChunk) {

            onChunk(
              chunk
            );

          }

        }

      }


      if (
        event.type ===
        "error"
      ) {

        throw new Error(
          event.message ||
          "Assistant streaming error."
        );

      }

    } catch (error) {

      if (
        error instanceof Error &&
        error.message !==
          "Unexpected end of JSON input"
      ) {

        throw error;

      }

    }

  }


  return {

    success:
      true,

    route,

    intent,

    response:
      fullResponse,

  };

}


/* =========================================================
   BACKEND STATUS
========================================================= */

export async function getBackendStatus():
  Promise<BackendStatus> {

  const response =
    await fetch(
      `${API_BASE}/status`
    );


  if (!response.ok) {

    throw new Error(
      `Status server error: ${response.status}`
    );

  }


  return response.json();

}


/* =========================================================
   OBJECT DETECTION
========================================================= */

export async function detectObjects(
  blob: Blob,
  confidence = 0.4
): Promise<{
  success: boolean;
  width?: number;
  height?: number;
  detections?: Detection[];
  object_count?: number;
  objects?: Record<string, number>;
  error?: string;
}> {

  const formData =
    new FormData();


  formData.append(
    "file",
    blob,
    "frame.jpg"
  );


  const response =
    await fetch(
      `${API_BASE}/detect?confidence=${confidence}`,
      {
        method: "POST",
        body: formData,
      }
    );


  if (!response.ok) {

    throw new Error(
      `Object detection server error: ${response.status}`
    );

  }


  return response.json();

}


/* =========================================================
   FACE RECOGNITION
========================================================= */

export async function recognizeFaces(
  blob: Blob
): Promise<{
  success: boolean;
  faces?: Face[];
  face_count?: number;
  database_people?: string[];
  error?: string;
}> {

  const formData =
    new FormData();


  formData.append(
    "file",
    blob,
    "frame.jpg"
  );


  const response =
    await fetch(
      `${API_BASE}/recognize`,
      {
        method: "POST",
        body: formData,
      }
    );


  if (!response.ok) {

    throw new Error(
      `Face recognition server error: ${response.status}`
    );

  }


  return response.json();

}


/* =========================================================
   FACE ENROLLMENT
========================================================= */

export async function enrollFace(
  blob: Blob,
  name: string
): Promise<EnrollResponse> {

  const formData =
    new FormData();


  formData.append(
    "file",
    blob,
    "face.jpg"
  );


  const response =
    await fetch(
      `${API_BASE}/enroll?name=${encodeURIComponent(
        name
      )}`,
      {
        method: "POST",
        body: formData,
      }
    );


  if (!response.ok) {

    throw new Error(
      `Enrollment server error: ${response.status}`
    );

  }


  const data =
    (await response.json()) as EnrollResponse;


  if (!data.success) {

    throw new Error(
      data.error ||
      "Face enrollment failed."
    );

  }


  return data;

}


/* =========================================================
   SPEECH TO TEXT
========================================================= */

export async function transcribeAudio(
  audioBlob: Blob
): Promise<string> {

  const formData =
    new FormData();


  formData.append(
    "file",
    audioBlob,
    "recording.webm"
  );


  const response =
    await fetch(
      `${API_BASE}/transcribe`,
      {
        method: "POST",
        body: formData,
      }
    );


  if (!response.ok) {

    throw new Error(
      `Transcription server error: ${response.status}`
    );

  }


  const data =
    (await response.json()) as TranscriptionResponse;


  return data.text || "";

}


/* =========================================================
   TEXT TO SPEECH
========================================================= */

export async function speakText(
  text: string
): Promise<HTMLAudioElement> {

  const response =
    await fetch(
      `${API_BASE}/speak`,
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body: JSON.stringify({
          text,
        }),
      }
    );


  if (!response.ok) {

    throw new Error(
      `TTS server error: ${response.status}`
    );

  }


  const audioBlob =
    await response.blob();


  const audioUrl =
    URL.createObjectURL(
      audioBlob
    );


  const audio =
    new Audio(
      audioUrl
    );


  const cleanup =
    () => {

      URL.revokeObjectURL(
        audioUrl
      );

    };


  audio.addEventListener(
    "ended",
    cleanup,
    {
      once: true,
    }
  );


  audio.addEventListener(
    "error",
    cleanup,
    {
      once: true,
    }
  );


  return audio;

}


/* =========================================================
   FILE READER
========================================================= */

export async function readFile(
  file: File
): Promise<FileReadResponse> {

  const formData =
    new FormData();


  formData.append(
    "file",
    file,
    file.name
  );


  const response =
    await fetch(
      `${API_BASE}/file/read`,
      {
        method: "POST",
        body: formData,
      }
    );


  if (!response.ok) {

    throw new Error(
      `File reader server error: ${response.status}`
    );

  }


  const data =
    (await response.json()) as FileReadResponse;


  if (!data.success) {

    throw new Error(
      data.error ||
      "Could not read file."
    );

  }


  return data;

}
