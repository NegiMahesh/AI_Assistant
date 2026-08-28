import type { BackendStatus } from "../services/api";

type Props = {
  backendConnection:
    | "checking"
    | "online"
    | "offline";

  backendStatus:
    | BackendStatus
    | null;

  onRetry: () => void;

  onClear: () => void;
};


export default function Header({
  backendConnection,
  backendStatus,
  onRetry,
  onClear,
}: Props) {

  const statusText =
    backendConnection ===
    "checking"
      ? "Checking backend..."
      : backendConnection ===
        "online"
        ? "Local AI · Online"
        : "Backend Offline";


  return (
    <header className="topbar">

      <div className="brand">

        <div className="brand-mark">
          <span className="brand-mark-core">
            ✦
          </span>
        </div>


        <div className="brand-copy">

          <div className="brand-name">
            FAST AI
          </div>

          <div className="brand-subtitle">
            Personal Intelligence Workspace
          </div>

        </div>

      </div>


      <div className="topbar-center">

        <div className="system-pill">

          <span
            className={
              `system-dot ${backendConnection}`
            }
          />

          <span>
            {statusText}
          </span>

        </div>


        {backendStatus?.llm && (

          <div className="model-pill">
            {backendStatus.llm}
          </div>

        )}

      </div>


      <div className="topbar-actions">

        {backendConnection ===
          "offline" && (

          <button
            className="topbar-button retry"
            type="button"
            onClick={onRetry}
          >
            ↻ Retry
          </button>

        )}


        <button
          className="topbar-button"
          type="button"
          onClick={onClear}
        >
          Clear
        </button>

      </div>

    </header>
  );
}