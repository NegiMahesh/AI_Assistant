
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
} from "react";

import "./App.css";

import {
  enrollFace,
  getBackendStatus,
  readFile,
  sendAssistantMessage,
  speakText,
  transcribeAudio,
  type BackendStatus,
  type ConversationMessage,
  type Detection,
  type Face,
} from "./services/api";

import Header from "./components/Header";
import ChatPanel from "./components/ChatPanel";
import CameraPanel from "./components/CameraPanel";
import FileBar from "./components/FileBar";
import VoiceStatus from "./components/VoiceStatus";
import EnrollModal from "./components/EnrollModal";


/* =========================================================
   TYPES
========================================================= */

export type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
};

export type CameraViewMode =
  | "compact"
  | "normal"
  | "large";

type BackendConnection =
  | "checking"
  | "online"
  | "offline";

type StoredConversation = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
};


/* =========================================================
   STORAGE
========================================================= */

const CONVERSATIONS_STORAGE_KEY =
  "fast-ai-conversations-v1";

const CURRENT_CONVERSATION_KEY =
  "fast-ai-current-conversation-v1";

const DEFAULT_MESSAGE: Message = {
  id: 1,
  role: "assistant",
  content:
    "Hello! I am your Fast AI Assistant. Ask me anything.",
};


/* =========================================================
   CONVERSATION HELPERS
========================================================= */

function createConversation(): StoredConversation {
  const now = Date.now();

  return {
    id:
      `${now}-${Math.random()
        .toString(36)
        .slice(2, 8)}`,

    title:
      "New Conversation",

    createdAt:
      now,

    updatedAt:
      now,

    messages: [
      {
        ...DEFAULT_MESSAGE,
        id: now,
      },
    ],
  };
}


function loadConversations(): StoredConversation[] {
  try {
    const raw =
      localStorage.getItem(
        CONVERSATIONS_STORAGE_KEY
      );

    if (!raw) {
      return [];
    }

    const parsed =
      JSON.parse(raw);

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.filter(
      (
        item: unknown
      ): item is StoredConversation => {
        if (
          !item ||
          typeof item !== "object"
        ) {
          return false;
        }

        const conversation =
          item as Partial<StoredConversation>;

        return (
          typeof conversation.id === "string" &&
          typeof conversation.title === "string" &&
          typeof conversation.createdAt === "number" &&
          typeof conversation.updatedAt === "number" &&
          Array.isArray(conversation.messages)
        );
      }
    );
  } catch (error) {
    console.error(
      "Conversation load error:",
      error
    );

    return [];
  }
}


function saveConversations(
  conversations: StoredConversation[]
) {
  try {
    localStorage.setItem(
      CONVERSATIONS_STORAGE_KEY,
      JSON.stringify(
        conversations
      )
    );
  } catch (error) {
    console.error(
      "Conversation save error:",
      error
    );
  }
}


function makeConversationTitle(
  messages: Message[]
): string {
  const firstUserMessage =
    messages.find(
      (message) =>
        message.role === "user" &&
        message.content.trim()
    );

  if (!firstUserMessage) {
    return "New Conversation";
  }

  const cleaned =
    firstUserMessage.content
      .replace(/\s+/g, " ")
      .trim();

  if (cleaned.length <= 32) {
    return cleaned;
  }

  return (
    cleaned
      .slice(0, 32)
      .trimEnd() +
    "..."
  );
}


function getInitialConversation(
  savedConversations: StoredConversation[]
): StoredConversation {
  if (
    savedConversations.length === 0
  ) {
    return createConversation();
  }

  let savedId: string | null = null;

  try {
    savedId =
      localStorage.getItem(
        CURRENT_CONVERSATION_KEY
      );
  } catch {
    savedId = null;
  }

  const savedConversation =
    savedConversations.find(
      (conversation) =>
        conversation.id === savedId
    );

  if (savedConversation) {
    return savedConversation;
  }

  const sorted =
    savedConversations
      .slice()
      .sort(
        (a, b) =>
          b.updatedAt -
          a.updatedAt
      );

  return sorted[0];
}


/* =========================================================
   APP
========================================================= */

function App() {

  /* =======================================================
     CONVERSATIONS
  ======================================================= */

  const [conversations, setConversations] =
    useState<StoredConversation[]>(() => {
      const saved =
        loadConversations();

      if (saved.length > 0) {
        return saved;
      }

      const fresh =
        createConversation();

      saveConversations([
        fresh,
      ]);

      return [
        fresh,
      ];
    });


  const [currentConversationId, setCurrentConversationId] =
    useState<string>(() => {
      const saved =
        loadConversations();

      if (saved.length === 0) {
        return "";
      }

      try {
        const savedId =
          localStorage.getItem(
            CURRENT_CONVERSATION_KEY
          );

        if (
          savedId &&
          saved.some(
            (conversation) =>
              conversation.id ===
              savedId
          )
        ) {
          return savedId;
        }
      } catch {
        /* Ignore storage errors. */
      }

      return getInitialConversation(
        saved
      ).id;
    });


  const [historyPanelOpen, setHistoryPanelOpen] =
    useState(false);


  /* =======================================================
     CURRENT CHAT
  ======================================================= */

  const initialConversation =
    getInitialConversation(
      conversations
    );


  const [messages, setMessages] =
    useState<Message[]>(
      initialConversation.messages
    );


  const [messageInput, setMessageInput] =
    useState("");


  const [isThinking, setIsThinking] =
    useState(false);


  /* =======================================================
     BACKEND
  ======================================================= */

  const [backendConnection, setBackendConnection] =
    useState<BackendConnection>(
      "checking"
    );


  const [backendStatus, setBackendStatus] =
    useState<BackendStatus | null>(
      null
    );


  /* =======================================================
     CAMERA
  ======================================================= */

  const [cameraRunning, setCameraRunning] =
    useState(false);


  const [cameraStatus, setCameraStatus] =
    useState(
      "Camera is off"
    );


  const [detections, setDetections] =
    useState<Detection[]>([]);


  const [faces, setFaces] =
    useState<Face[]>([]);


  const [detectionFPS, setDetectionFPS] =
    useState(0);


  const [cameraViewMode, setCameraViewMode] =
    useState<CameraViewMode>(
      "normal"
    );


  /* =======================================================
     VOICE
  ======================================================= */

  const [isRecording, setIsRecording] =
    useState(false);


  const [voiceStatus, setVoiceStatus] =
    useState("");


  const [isTranscribing, setIsTranscribing] =
    useState(false);


  /* =======================================================
     TTS
  ======================================================= */

  const [speakingMessageId, setSpeakingMessageId] =
    useState<number | null>(
      null
    );


  /* =======================================================
     FILE
  ======================================================= */

  const [fileName, setFileName] =
    useState("");


  const [fileContext, setFileContext] =
    useState("");


  const [fileCharacters, setFileCharacters] =
    useState(0);


  const [fileLoading, setFileLoading] =
    useState(false);


  /* =======================================================
     ENROLLMENT
  ======================================================= */

  const [enrollPanelOpen, setEnrollPanelOpen] =
    useState(false);


  const [enrollName, setEnrollName] =
    useState("");


  const [enrollStatus, setEnrollStatus] =
    useState("");


  const [isEnrolling, setIsEnrolling] =
    useState(false);


  /* =======================================================
     REFS
  ======================================================= */

  const mediaRecorderRef =
    useRef<MediaRecorder | null>(
      null
    );


  const audioChunksRef =
    useRef<Blob[]>([]);


  const currentAudioRef =
    useRef<HTMLAudioElement | null>(
      null
    );


  const fileInputRef =
    useRef<HTMLInputElement | null>(
      null
    );


  const requestControllerRef =
    useRef<AbortController | null>(
      null
    );


  /* =======================================================
     CURRENT CONVERSATION FIX
  ======================================================= */

  useEffect(() => {
    const exists =
      conversations.some(
        (conversation) =>
          conversation.id ===
          currentConversationId
      );

    if (
      !currentConversationId ||
      !exists
    ) {
      setCurrentConversationId(
        getInitialConversation(
          conversations
        ).id
      );
    }
  }, [
    conversations,
    currentConversationId,
  ]);


  /* =======================================================
     SAVE CURRENT CONVERSATION
  ======================================================= */

  useEffect(() => {
    if (!currentConversationId) {
      return;
    }

    setConversations(
      (current) => {
        const exists =
          current.some(
            (conversation) =>
              conversation.id ===
              currentConversationId
          );

        if (!exists) {
          return current;
        }

        const updated =
          current.map(
            (conversation) => {
              if (
                conversation.id !==
                currentConversationId
              ) {
                return conversation;
              }

              return {
                ...conversation,

                title:
                  makeConversationTitle(
                    messages
                  ),

                updatedAt:
                  Date.now(),

                messages:
                  messages.map(
                    (message) => ({
                      ...message,
                    })
                  ),
              };
            }
          );

        saveConversations(
          updated
        );

        return updated;
      }
    );
  }, [
    messages,
    currentConversationId,
  ]);


  /* =======================================================
     SAVE CURRENT CONVERSATION ID
  ======================================================= */

  useEffect(() => {
    if (!currentConversationId) {
      return;
    }

    try {
      localStorage.setItem(
        CURRENT_CONVERSATION_KEY,
        currentConversationId
      );
    } catch (error) {
      console.error(
        "Current conversation save error:",
        error
      );
    }
  }, [
    currentConversationId,
  ]);


  /* =======================================================
     BACKEND STATUS
  ======================================================= */

  const checkBackend =
    useCallback(
      async () => {
        setBackendConnection(
          "checking"
        );

        try {
          const data =
            await getBackendStatus();

          setBackendStatus(
            data
          );

          setBackendConnection(
            "online"
          );

        } catch (error) {
          console.error(
            "Backend status error:",
            error
          );

          setBackendStatus(
            null
          );

          setBackendConnection(
            "offline"
          );
        }
      },
      []
    );


  useEffect(() => {
    void checkBackend();

    const timer =
      window.setInterval(
        () => {
          void checkBackend();
        },
        10000
      );

    return () => {
      window.clearInterval(
        timer
      );
    };
  }, [
    checkBackend,
  ]);


  /* =======================================================
     NEW CHAT
  ======================================================= */

  function createNewChat() {
    if (isThinking) {
      requestControllerRef.current?.abort();
    }

    const conversation =
      createConversation();

    setConversations(
      (current) => {
        const updated = [
          conversation,
          ...current,
        ];

        saveConversations(
          updated
        );

        return updated;
      }
    );

    setCurrentConversationId(
      conversation.id
    );

    setMessages(
      conversation.messages
    );

    setMessageInput("");

    setFileName("");

    setFileContext("");

    setFileCharacters(0);

    setHistoryPanelOpen(
      false
    );
  }


  /* =======================================================
     OPEN SAVED CHAT
  ======================================================= */

  function openConversation(
    conversationId: string
  ) {
    if (isThinking) {
      requestControllerRef.current?.abort();

      setIsThinking(
        false
      );
    }

    const conversation =
      conversations.find(
        (item) =>
          item.id ===
          conversationId
      );

    if (!conversation) {
      return;
    }

    setCurrentConversationId(
      conversation.id
    );

    setMessages(
      conversation.messages
    );

    setMessageInput("");

    setFileName("");

    setFileContext("");

    setFileCharacters(0);

    setHistoryPanelOpen(
      false
    );
  }


  /* =======================================================
     DELETE CHAT
  ======================================================= */

  function deleteConversation(
    conversationId: string
  ) {
    if (
      isThinking &&
      conversationId ===
        currentConversationId
    ) {
      requestControllerRef.current?.abort();

      setIsThinking(
        false
      );
    }

    const remaining =
      conversations.filter(
        (conversation) =>
          conversation.id !==
          conversationId
      );

    if (
      remaining.length ===
      0
    ) {
      const fresh =
        createConversation();

      const updated = [
        fresh,
      ];

      saveConversations(
        updated
      );

      setConversations(
        updated
      );

      setCurrentConversationId(
        fresh.id
      );

      setMessages(
        fresh.messages
      );

      setMessageInput("");

      setFileName("");

      setFileContext("");

      setFileCharacters(0);

      return;
    }

    saveConversations(
      remaining
    );

    setConversations(
      remaining
    );

    if (
      currentConversationId ===
      conversationId
    ) {
      const sorted =
        remaining
          .slice()
          .sort(
            (a, b) =>
              b.updatedAt -
              a.updatedAt
          );

      const next =
        sorted[0];

      setCurrentConversationId(
        next.id
      );

      setMessages(
        next.messages
      );

      setMessageInput("");

      setFileName("");

      setFileContext("");

      setFileCharacters(0);
    }
  }


  /* =======================================================
     CLEAR CURRENT CHAT
  ======================================================= */

  function clearCurrentChat() {
    if (isThinking) {
      requestControllerRef.current?.abort();

      setIsThinking(
        false
      );
    }

    const freshMessage: Message = {
      id:
        Date.now(),

      role:
        "assistant",

      content:
        "New conversation started. How can I help?",
    };

    setMessages([
      freshMessage,
    ]);

    setMessageInput("");

    setFileName("");

    setFileContext("");

    setFileCharacters(0);

    setConversations(
      (current) => {
        const updated =
          current.map(
            (conversation) => {
              if (
                conversation.id !==
                currentConversationId
              ) {
                return conversation;
              }

              return {
                ...conversation,

                title:
                  "New Conversation",

                updatedAt:
                  Date.now(),

                messages: [
                  freshMessage,
                ],
              };
            }
          );

        saveConversations(
          updated
        );

        return updated;
      }
    );
  }


  /* =======================================================
     SEND MESSAGE
  ======================================================= */

  async function handleSend() {
    const text =
      messageInput.trim();

    if (
      !text ||
      isThinking ||
      fileLoading ||
      isTranscribing
    ) {
      return;
    }

    if (
      backendConnection !==
      "online"
    ) {
      setMessages(
        (current) => [
          ...current,
          {
            id:
              Date.now(),

            role:
              "assistant",

            content:
              "❌ The FastAPI backend is offline. Start the backend and try again.",
          },
        ]
      );

      return;
    }

    const history:
      ConversationMessage[] =
      messages
        .filter(
          (message) =>
            message.content.trim()
        )
        .slice(-12)
        .map(
          (message) => ({
            role:
              message.role,

            content:
              message.content,
          })
        );

    const userMessage: Message = {
      id:
        Date.now(),

      role:
        "user",

      content:
        text,
    };

    const thinkingId =
      Date.now() + 1;

    setMessages(
      (current) => [
        ...current,

        userMessage,

        {
          id:
            thinkingId,

          role:
            "assistant",

          content:
            "",
        },
      ]
    );

    setMessageInput("");

    setIsThinking(
      true
    );

    const controller =
      new AbortController();

    requestControllerRef.current =
      controller;

    try {
      const response =
        await sendAssistantMessage(
          text,

          {
            detections,
            faces,
          },

          fileContext ||
            null,

          history,

          controller.signal,

          (chunk) => {
            setMessages(
              (current) =>
                current.map(
                  (message) =>
                    message.id ===
                    thinkingId
                      ? {
                          ...message,

                          content:
                            message.content +
                            chunk,
                        }
                      : message
                )
            );
          }
        );

      if (
        response.response &&
        response.response.trim()
      ) {
        setMessages(
          (current) =>
            current.map(
              (message) =>
                message.id ===
                thinkingId
                  ? {
                      ...message,

                      content:
                        response.response ||
                        "No response received.",
                    }
                  : message
            )
        );
      }
    } catch (error) {
      if (
        error instanceof DOMException &&
        error.name ===
          "AbortError"
      ) {
        setMessages(
          (current) =>
            current.map(
              (message) =>
                message.id ===
                thinkingId
                  ? {
                      ...message,

                      content:
                        message.content.trim()
                          ? message.content +
                            "\n\n⏹ Generation stopped."
                          : "⏹ Generation stopped.",
                    }
                  : message
            )
        );

        return;
      }

      console.error(
        "Assistant error:",
        error
      );

      const errorText =
        error instanceof Error
          ? error.message
          : "Unknown error";

      setMessages(
        (current) =>
          current.map(
            (message) =>
              message.id ===
              thinkingId
                ? {
                    ...message,

                    content:
                      `❌ ${errorText}`,
                  }
                : message
          )
      );

      void checkBackend();
    } finally {
      setIsThinking(
        false
      );

      requestControllerRef.current =
        null;
    }
  }


  /* =======================================================
     STOP GENERATION
  ======================================================= */

  function stopGeneration() {
    requestControllerRef.current?.abort();
  }


  /* =======================================================
     VOICE RECORDING
  ======================================================= */

  async function startRecording() {
    if (
      isRecording ||
      isThinking ||
      isTranscribing ||
      fileLoading
    ) {
      return;
    }

    try {
      const stream =
        await navigator
          .mediaDevices
          .getUserMedia({
            audio: true,
          });

      let mimeType =
        "audio/webm";

      if (
        !MediaRecorder.isTypeSupported(
          mimeType
        )
      ) {
        mimeType =
          "audio/webm;codecs=opus";
      }

      const recorder =
        new MediaRecorder(
          stream,
          {
            mimeType,
          }
        );

      audioChunksRef.current =
        [];

      recorder.ondataavailable =
        (event) => {
          if (
            event.data.size >
            0
          ) {
            audioChunksRef.current.push(
              event.data
            );
          }
        };

      recorder.onstop =
        async () => {
          stream
            .getTracks()
            .forEach(
              (track) =>
                track.stop()
            );

          const blob =
            new Blob(
              audioChunksRef.current,
              {
                type:
                  recorder.mimeType ||
                  "audio/webm",
              }
            );

          setIsTranscribing(
            true
          );

          setVoiceStatus(
            "🧠 Converting speech to text..."
          );

          try {
            const text =
              await transcribeAudio(
                blob
              );

            if (
              text.trim()
            ) {
              setMessageInput(
                text.trim()
              );

              setVoiceStatus(
                "✅ Speech converted"
              );
            } else {
              setVoiceStatus(
                "⚠️ No speech detected"
              );
            }

            window.setTimeout(
              () => {
                setVoiceStatus(
                  ""
                );
              },
              1800
            );
          } catch (error) {
            console.error(
              "Transcription error:",
              error
            );

            setVoiceStatus(
              "❌ Transcription failed"
            );

            window.setTimeout(
              () => {
                setVoiceStatus(
                  ""
                );
              },
              2200
            );
          } finally {
            setIsTranscribing(
              false
            );
          }
        };

      recorder.start();

      mediaRecorderRef.current =
        recorder;

      setIsRecording(
        true
      );

      setVoiceStatus(
        "🎤 Listening..."
      );
    } catch (error) {
      console.error(
        "Microphone error:",
        error
      );

      alert(
        "Microphone access failed. Please allow microphone permission."
      );
    }
  }


  /* =======================================================
     STOP RECORDING
  ======================================================= */

  function stopRecording() {
    const recorder =
      mediaRecorderRef.current;

    if (!recorder) {
      return;
    }

    if (
      recorder.state !==
      "inactive"
    ) {
      recorder.stop();
    }

    mediaRecorderRef.current =
      null;

    setIsRecording(
      false
    );

    setVoiceStatus(
      "🧠 Converting speech to text..."
    );
  }


  /* =======================================================
     MANUAL VOICE BUTTON
  ======================================================= */

  function handleVoiceClick() {
    if (isRecording) {
      stopRecording();
    } else {
      void startRecording();
    }
  }


  /* =======================================================
     TEXT TO SPEECH
  ======================================================= */

  async function handleSpeak(
    messageId: number,
    text: string
  ) {
    if (!text.trim()) {
      return;
    }

    if (
      speakingMessageId ===
      messageId
    ) {
      currentAudioRef.current?.pause();

      currentAudioRef.current =
        null;

      setSpeakingMessageId(
        null
      );

      return;
    }

    currentAudioRef.current?.pause();

    currentAudioRef.current =
      null;

    setSpeakingMessageId(
      messageId
    );

    try {
      const audio =
        await speakText(
          text
        );

      currentAudioRef.current =
        audio;

      audio.onended =
        () => {
          setSpeakingMessageId(
            null
          );

          currentAudioRef.current =
            null;
        };

      audio.onerror =
        () => {
          setSpeakingMessageId(
            null
          );

          currentAudioRef.current =
            null;
        };

      await audio.play();
    } catch (error) {
      console.error(
        "TTS error:",
        error
      );

      setSpeakingMessageId(
        null
      );

      currentAudioRef.current =
        null;
    }
  }


  /* =======================================================
     FILE PICKER
  ======================================================= */

  function openFilePicker() {
    if (
      fileLoading ||
      isThinking
    ) {
      return;
    }

    fileInputRef.current?.click();
  }


  /* =======================================================
     FILE SELECTED
  ======================================================= */

  async function handleFileSelected(
    event: ChangeEvent<HTMLInputElement>
  ) {
    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    const extension =
      "." +
      (
        file.name
          .split(".")
          .pop() ||
        ""
      ).toLowerCase();

    const allowed = [
      ".txt",
      ".pdf",
      ".docx",
    ];

    if (
      !allowed.includes(
        extension
      )
    ) {
      alert(
        "Supported files: TXT, PDF and DOCX"
      );

      event.target.value =
        "";

      return;
    }

    setFileLoading(
      true
    );

    setFileName(
      file.name
    );

    setFileContext(
      ""
    );

    setFileCharacters(
      0
    );

    try {
      const data =
        await readFile(
          file
        );

      if (
        !data.text?.trim()
      ) {
        throw new Error(
          "No readable text was found in this file."
        );
      }

      setFileContext(
        data.text
      );

      setFileCharacters(
        data.characters ||
        data.text.length
      );

      setMessages(
        (current) => [
          ...current,
          {
            id:
              Date.now(),

            role:
              "assistant",

            content:
              `📎 ${
                data.filename ||
                file.name
              } is ready.\n\nAsk me anything about this file.`,
          },
        ]
      );
    } catch (error) {
      console.error(
        "File read error:",
        error
      );

      const errorText =
        error instanceof Error
          ? error.message
          : "Unable to read file.";

      setFileName("");

      setFileContext("");

      setFileCharacters(
        0
      );

      setMessages(
        (current) => [
          ...current,
          {
            id:
              Date.now(),

            role:
              "assistant",

            content:
              `❌ File error: ${errorText}`,
          },
        ]
      );
    } finally {
      setFileLoading(
        false
      );

      event.target.value =
        "";
    }
  }


  /* =======================================================
     REMOVE FILE
  ======================================================= */

  function removeFile() {
    setFileName("");

    setFileContext("");

    setFileCharacters(
      0
    );

    if (
      fileInputRef.current
    ) {
      fileInputRef.current.value =
        "";
    }
  }


  /* =======================================================
     ENROLLMENT
  ======================================================= */

  function openEnrollPanel() {
    if (
      !cameraRunning
    ) {
      alert(
        "Start the camera before enrolling a face."
      );

      return;
    }

    setEnrollName("");

    setEnrollStatus("");

    setEnrollPanelOpen(
      true
    );
  }


  function closeEnrollPanel() {
    if (
      isEnrolling
    ) {
      return;
    }

    setEnrollPanelOpen(
      false
    );
  }


  async function handleEnroll() {
    const name =
      enrollName.trim();

    if (!name) {
      setEnrollStatus(
        "❌ Enter your name."
      );

      return;
    }

    const video =
      document.querySelector(
        ".camera-video"
      ) as HTMLVideoElement |
      null;

    if (
      !video ||
      video.readyState <
        2
    ) {
      setEnrollStatus(
        "❌ Camera is not ready."
      );

      return;
    }

    setIsEnrolling(
      true
    );

    setEnrollStatus(
      "🧠 Capturing face..."
    );

    try {
      const canvas =
        document.createElement(
          "canvas"
        );

      canvas.width =
        video.videoWidth;

      canvas.height =
        video.videoHeight;

      const context =
        canvas.getContext(
          "2d"
        );

      if (!context) {
        throw new Error(
          "Could not capture camera frame."
        );
      }

      context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
      );

      const blob =
        await new Promise<
          Blob | null
        >(
          (resolve) =>
            canvas.toBlob(
              resolve,
              "image/jpeg",
              0.85
            )
        );

      if (!blob) {
        throw new Error(
          "Could not create image."
        );
      }

      setEnrollStatus(
        "🧠 Processing face..."
      );

      const result =
        await enrollFace(
          blob,
          name
        );

      setEnrollStatus(
        `✅ ${
          result.message ||
          `${name} was enrolled successfully.`
        }`
      );

      setCameraStatus(
        `✅ ${name} enrolled successfully`
      );

      window.setTimeout(
        () => {
          setEnrollPanelOpen(
            false
          );

          setEnrollStatus(
            ""
          );
        },
        1500
      );
    } catch (error) {
      console.error(
        "Enrollment error:",
        error
      );

      setEnrollStatus(
        `❌ ${
          error instanceof Error
            ? error.message
            : "Enrollment failed."
        }`
      );
    } finally {
      setIsEnrolling(
        false
      );
    }
  }


  function handleEnrollKeyDown(
    event: KeyboardEvent<HTMLInputElement>
  ) {
    if (
      event.key ===
        "Enter" &&
      !isEnrolling
    ) {
      event.preventDefault();

      void handleEnroll();
    }
  }


  /* =======================================================
     CLEANUP
  ======================================================= */

  useEffect(() => {
    return () => {
      requestControllerRef.current?.abort();

      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !==
          "inactive"
      ) {
        mediaRecorderRef.current.stop();
      }

      currentAudioRef.current?.pause();
    };
  }, []);


  /* =======================================================
     HISTORY
  ======================================================= */

  const sortedConversations =
    conversations
      .slice()
      .sort(
        (a, b) =>
          b.updatedAt -
          a.updatedAt
      );


  const currentConversation =
    conversations.find(
      (conversation) =>
        conversation.id ===
        currentConversationId
    );


  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div className="app">

      <Header
        backendConnection={
          backendConnection
        }

        backendStatus={
          backendStatus
        }

        onRetry={() =>
          void checkBackend()
        }

        onClear={
          clearCurrentChat
        }
      />


      {/* =================================================
         HISTORY TOOLBAR
      ================================================= */}

      <div className="history-toolbar">

        <button
          className="history-button"
          type="button"
          onClick={() =>
            setHistoryPanelOpen(
              (open) =>
                !open
            )
          }
        >
          {
            historyPanelOpen
              ? "✕ Close"
              : "☰ History"
          }
        </button>


        <button
          className="new-chat-button"
          type="button"
          onClick={
            createNewChat
          }
        >
          ＋ New Chat
        </button>


        <div className="current-chat-title">
          {
            currentConversation?.title ||
            "New Conversation"
          }
        </div>

      </div>


      {/* =================================================
         HISTORY PANEL
      ================================================= */}

      {historyPanelOpen && (
        <aside
          className="history-panel"
        >

          <div
            className="history-panel-header"
          >

            <span>
              Conversations
            </span>

            <span className="history-count">
              {
                conversations.length
              }
            </span>

          </div>


          <div className="history-list">

            {sortedConversations.map(
              (
                conversation
              ) => (

                <div
                  className={
                    `history-item ${
                      conversation.id ===
                      currentConversationId
                        ? "active"
                        : ""
                    }`
                  }

                  key={
                    conversation.id
                  }
                >

                  <button
                    className="history-open"
                    type="button"
                    onClick={() =>
                      openConversation(
                        conversation.id
                      )
                    }
                  >

                    <span className="history-item-title">
                      {
                        conversation.title
                      }
                    </span>


                    <span className="history-item-date">
                      {
                        new Date(
                          conversation.updatedAt
                        ).toLocaleString()
                      }
                    </span>

                  </button>


                  <button
                    className="history-delete"
                    type="button"
                    title="Delete conversation"
                    onClick={() =>
                      deleteConversation(
                        conversation.id
                      )
                    }
                  >
                    🗑
                  </button>

                </div>
              )
            )}

          </div>

        </aside>
      )}


      {/* =================================================
         WORKSPACE
      ================================================= */}

      <main
        className={
          `workspace workspace-${cameraViewMode}`
        }
      >

        <ChatPanel
          messages={
            messages
          }

          speakingMessageId={
            speakingMessageId
          }

          onSpeak={
            handleSpeak
          }
        />


        <CameraPanel
          cameraRunning={
            cameraRunning
          }

          setCameraRunning={
            setCameraRunning
          }

          cameraStatus={
            cameraStatus
          }

          setCameraStatus={
            setCameraStatus
          }

          detections={
            detections
          }

          setDetections={
            setDetections
          }

          faces={
            faces
          }

          setFaces={
            setFaces
          }

          detectionFPS={
            detectionFPS
          }

          setDetectionFPS={
            setDetectionFPS
          }

          cameraViewMode={
            cameraViewMode
          }

          setCameraViewMode={
            setCameraViewMode
          }

          onEnroll={
            openEnrollPanel
          }
        />

      </main>


      {/* =================================================
         FILE BAR
      ================================================= */}

      <FileBar
        fileName={
          fileName
        }

        characters={
          fileCharacters
        }

        loading={
          fileLoading
        }

        onRemove={
          removeFile
        }
      />


      {/* =================================================
         VOICE STATUS
      ================================================= */}

      <VoiceStatus
        text={
          voiceStatus
        }
      />


      {/* =================================================
         ENROLL MODAL
      ================================================= */}

      {enrollPanelOpen && (
        <EnrollModal

          open={
            enrollPanelOpen
          }

          name={
            enrollName
          }

          status={
            enrollStatus
          }

          loading={
            isEnrolling
          }

          onNameChange={
            setEnrollName
          }

          onKeyDown={
            handleEnrollKeyDown
          }

          onEnroll={() =>
            void handleEnroll()
          }

          onClose={
            closeEnrollPanel
          }
        />
      )}


      {/* =================================================
         COMPOSER
      ================================================= */}

      <footer
        className="composer"
      >

        <div
          className="composer-inner"
        >

          {/* FILE BUTTON */}

          <button
            className="composer-icon"
            type="button"
            title="Upload TXT, PDF or DOCX"
            onClick={
              openFilePicker
            }

            disabled={
              fileLoading ||
              isThinking
            }
          >
            {
              fileLoading
                ? "⏳"
                : "📎"
            }
          </button>


          <input
            ref={
              fileInputRef
            }

            type="file"

            accept=".txt,.pdf,.docx"

            hidden

            onChange={
              handleFileSelected
            }
          />


          {/* MICROPHONE */}

          <button
            className={
              `composer-icon mic-control ${
                isRecording
                  ? "recording"
                  : ""
              }`
            }

            type="button"

            title={
              isRecording
                ? "Stop recording"
                : "Voice input"
            }

            onClick={
              handleVoiceClick
            }

            disabled={
              isThinking ||
              isTranscribing ||
              fileLoading
            }
          >

            {
              isRecording
                ? "⏹"
                : "🎤"
            }

          </button>


          {/* TEXT INPUT */}

          <input
            className="composer-input"

            type="text"

            value={
              messageInput
            }

            placeholder={
              fileName
                ? "Ask about your file..."
                : "Ask me anything..."
            }

            onChange={
              (event) =>
                setMessageInput(
                  event.target.value
                )
            }

            onKeyDown={
              (event) => {
                if (
                  event.key ===
                    "Enter" &&
                  !event.shiftKey &&
                  !isThinking
                ) {
                  event.preventDefault();

                  void handleSend();
                }
              }
            }

            disabled={
              isTranscribing ||
              fileLoading
            }
          />


          {/* =================================================
             STOP / SEND
          ================================================= */}

          {isThinking ? (
            <button
              className="composer-stop"
              type="button"
              title="Stop generation"
              onClick={
                stopGeneration
              }
            >
              ■
            </button>
          ) : (
            <button
              className="composer-send"
              type="button"
              title="Send"
              onClick={() =>
                void handleSend()
              }

              disabled={
                isTranscribing ||
                fileLoading ||
                !messageInput.trim()
              }
            >
              ➤
            </button>
          )}

        </div>


        {/* COMPOSER STATUS */}

        <div
          className="composer-hint"
        >
          {
            isThinking

              ? "Generating response · Click ■ to stop"

              : isRecording

                ? "🎤 Listening..."

                : backendConnection ===
                  "offline"

                  ? "Backend offline · Start FastAPI to continue"

                  : fileName

                    ? "File attached · Questions are answered using the uploaded content"

                    : "Local AI · Streaming · Persistent History · Vision · Voice · Memory"
          }
        </div>

      </footer>

    </div>
  );
}


export default App;
