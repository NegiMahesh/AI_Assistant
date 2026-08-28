import type { Message } from "../App";

type Props = {
  message: Message;

  speaking:
    boolean;

  onSpeak:
    (messageId: number, text: string) => void;
};


export default function MessageBubble({
  message,
  speaking,
  onSpeak,
}: Props) {

  const isAssistant =
    message.role ===
    "assistant";


  return (
    <div
      className={
        `message-row ${
          isAssistant
            ? "assistant-message"
            : "user-message"
        }`
      }
    >

      {isAssistant && (

        <div className="message-avatar assistant-avatar">
          ✦
        </div>

      )}


      <div
        className={
          `message-card ${
            isAssistant
              ? "assistant-card"
              : "user-card"
          }`
        }
      >

        <div className="message-meta">

          <span>

            {
              isAssistant
                ? "FAST AI"
                : "YOU"
            }

          </span>

        </div>


        <div className="message-text">
          {
            message.content
          }
        </div>


        {isAssistant && (

          <button
            className={
              `message-speak ${
                speaking
                  ? "active"
                  : ""
              }`
            }
            type="button"
            title={
              speaking
                ? "Stop speaking"
                : "Speak response"
            }
            onClick={() =>
              onSpeak(
                message.id,
                message.content
              )
            }
          >

            {
              speaking
                ? "⏹"
                : "🔊"
            }

          </button>

        )}

      </div>


      {!isAssistant && (

        <div className="message-avatar user-avatar">
          YOU
        </div>

      )}

    </div>
  );
}