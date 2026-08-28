import type { Message } from "../App";

import MessageBubble from "./MessageBubble";

type Props = {
  messages:
    Message[];

  speakingMessageId:
    number | null;

  onSpeak:
    (messageId: number, text: string) => void;
};


export default function ChatPanel({
  messages,
  speakingMessageId,
  onSpeak,
}: Props) {

  return (
    <section className="chat-panel">

      <div className="section-heading">

        <div className="section-heading-left">

          <span className="heading-icon">
            ◈
          </span>

          <div>

            <div className="section-title">
              Conversation
            </div>

            <div className="section-caption">
              Your local AI workspace
            </div>

          </div>

        </div>


        <div className="message-count">
          {messages.length}
        </div>

      </div>


      <div className="chat-scroll">

        {messages.length ===
          0 ? (

          <div className="empty-state">

            <div className="empty-orb">
              ✦
            </div>

            <h2>
              Start a conversation
            </h2>

            <p>
              Ask a question, use your microphone,
              upload a file, or activate vision.
            </p>

          </div>

        ) : (

          messages.map(
            (message) => (

              <MessageBubble
                key={
                  message.id
                }
                message={
                  message
                }
                speaking={
                  speakingMessageId ===
                  message.id
                }
                onSpeak={
                  onSpeak
                }
              />

            )
          )

        )}

      </div>

    </section>
  );
}