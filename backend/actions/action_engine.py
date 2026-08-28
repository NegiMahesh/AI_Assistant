
import os
import platform
import subprocess
import webbrowser
from urllib.parse import quote


# =========================================================
# ACTION RESULT
# =========================================================

def action_result(
    success: bool,
    action: str,
    message: str,
    error: str | None = None
):

    result = {
        "success": success,
        "action": action,
        "message": message,
    }

    if error:
        result["error"] = error

    return result


# =========================================================
# NORMALIZE
# =========================================================

def normalize_command(
    text: str
):

    return (
        text
        .strip()
        .lower()
    )


# =========================================================
# EXTRACT TARGET
# =========================================================

def extract_target(
    text: str,
    prefixes
):

    cleaned = text.strip()

    for prefix in prefixes:

        if cleaned.lower().startswith(prefix):

            return (
                cleaned[
                    len(prefix):
                ]
                .strip()
            )

    return cleaned


# =========================================================
# WEBSITE ALIASES
# =========================================================

WEBSITE_ALIASES = {

    "youtube":
        "https://www.youtube.com",

    "youtube.com":
        "https://www.youtube.com",

    "google":
        "https://www.google.com",

    "google.com":
        "https://www.google.com",

    "github":
        "https://github.com",

    "github.com":
        "https://github.com",

    "gmail":
        "https://mail.google.com",

    "gmail.com":
        "https://mail.google.com",

    "chatgpt":
        "https://chatgpt.com",

    "chatgpt.com":
        "https://chatgpt.com",

    "facebook":
        "https://www.facebook.com",

    "facebook.com":
        "https://www.facebook.com",

    "instagram":
        "https://www.instagram.com",

    "instagram.com":
        "https://www.instagram.com",

    "linkedin":
        "https://www.linkedin.com",

    "linkedin.com":
        "https://www.linkedin.com",

    "whatsapp":
        "https://web.whatsapp.com",

    "whatsapp web":
        "https://web.whatsapp.com",

}


# =========================================================
# OPEN WEBSITE
# =========================================================

def open_website(
    target: str
):

    target = (
        target
        .strip()
        .strip("\"'")
    )

    if not target:

        return action_result(
            False,
            "open_website",
            "Please specify a website."
        )


    normalized = (
        target
        .lower()
        .strip()
    )


    # -----------------------------------------------------
    # DIRECT ALIAS
    # -----------------------------------------------------

    if normalized in WEBSITE_ALIASES:

        url = WEBSITE_ALIASES[
            normalized
        ]

        display_name = target

    # -----------------------------------------------------
    # FULL URL
    # -----------------------------------------------------

    elif (
        normalized.startswith(
            "http://"
        )
        or
        normalized.startswith(
            "https://"
        )
    ):

        url = target

        display_name = target

    # -----------------------------------------------------
    # DOMAIN
    # -----------------------------------------------------

    elif "." in normalized:

        url = (
            "https://"
            + normalized
        )

        display_name = target

    # -----------------------------------------------------
    # UNKNOWN WORD
    # -----------------------------------------------------

    else:

        url = (
            "https://www.google.com/search?q="
            + quote(target)
        )

        display_name = target


    try:

        opened = webbrowser.open(
            url
        )

        if not opened:

            return action_result(
                False,
                "open_website",
                f"I couldn't open {display_name}."
            )


        return action_result(
            True,
            "open_website",
            f"Opening {display_name}."
        )


    except Exception as error:

        return action_result(
            False,
            "open_website",
            f"I couldn't open {display_name}.",
            str(error)
        )


# =========================================================
# OPEN APPLICATION
# =========================================================

def open_application(
    app_name: str
):

    app_name = (
        app_name
        .lower()
        .strip()
    )

    system = platform.system()


    # -----------------------------------------------------
    # CHROME
    # -----------------------------------------------------

    if (
        "chrome" in app_name
        or "google chrome" in app_name
    ):

        if system == "Windows":

            try:

                subprocess.Popen(
                    [
                        "cmd",
                        "/c",
                        "start",
                        "",
                        "chrome"
                    ],
                    shell=False
                )

                return action_result(
                    True,
                    "open_app",
                    "Opening Chrome."
                )

            except Exception as error:

                return action_result(
                    False,
                    "open_app",
                    "I couldn't open Chrome.",
                    str(error)
                )


    # -----------------------------------------------------
    # CALCULATOR
    # -----------------------------------------------------

    if (
        "calculator" in app_name
        or app_name == "calc"
    ):

        if system == "Windows":

            try:

                subprocess.Popen(
                    "calc.exe"
                )

                return action_result(
                    True,
                    "open_app",
                    "Opening Calculator."
                )

            except Exception as error:

                return action_result(
                    False,
                    "open_app",
                    "I couldn't open Calculator.",
                    str(error)
                )


    # -----------------------------------------------------
    # NOTEPAD
    # -----------------------------------------------------

    if "notepad" in app_name:

        if system == "Windows":

            try:

                subprocess.Popen(
                    "notepad.exe"
                )

                return action_result(
                    True,
                    "open_app",
                    "Opening Notepad."
                )

            except Exception as error:

                return action_result(
                    False,
                    "open_app",
                    "I couldn't open Notepad.",
                    str(error)
                )


    # -----------------------------------------------------
    # PAINT
    # -----------------------------------------------------

    if (
        "paint" in app_name
        or "mspaint" in app_name
    ):

        if system == "Windows":

            try:

                subprocess.Popen(
                    "mspaint.exe"
                )

                return action_result(
                    True,
                    "open_app",
                    "Opening Paint."
                )

            except Exception as error:

                return action_result(
                    False,
                    "open_app",
                    "I couldn't open Paint.",
                    str(error)
                )


    # -----------------------------------------------------
    # FILE EXPLORER
    # -----------------------------------------------------

    if (
        "file explorer" in app_name
        or "explorer" in app_name
    ):

        if system == "Windows":

            try:

                subprocess.Popen(
                    "explorer.exe"
                )

                return action_result(
                    True,
                    "open_app",
                    "Opening File Explorer."
                )

            except Exception as error:

                return action_result(
                    False,
                    "open_app",
                    "I couldn't open File Explorer.",
                    str(error)
                )


    # -----------------------------------------------------
    # COMMAND PROMPT
    # -----------------------------------------------------

    if (
        "command prompt" in app_name
        or app_name == "cmd"
    ):

        if system == "Windows":

            try:

                subprocess.Popen(
                    "cmd.exe"
                )

                return action_result(
                    True,
                    "open_app",
                    "Opening Command Prompt."
                )

            except Exception as error:

                return action_result(
                    False,
                    "open_app",
                    "I couldn't open Command Prompt.",
                    str(error)
                )


    # -----------------------------------------------------
    # POWERSHELL
    # -----------------------------------------------------

    if (
        "powershell" in app_name
        or "power shell" in app_name
    ):

        if system == "Windows":

            try:

                subprocess.Popen(
                    "powershell.exe"
                )

                return action_result(
                    True,
                    "open_app",
                    "Opening PowerShell."
                )

            except Exception as error:

                return action_result(
                    False,
                    "open_app",
                    "I couldn't open PowerShell.",
                    str(error)
                )


    return action_result(
        False,
        "open_app",
        f"I don't have permission to open '{app_name}'. "
        "That application is not in my safe action list."
    )


# =========================================================
# CLOSE APPLICATION
# =========================================================

def close_application(
    app_name: str
):

    app_name = (
        app_name
        .lower()
        .strip()
    )

    system = platform.system()


    if system != "Windows":

        return action_result(
            False,
            "close_app",
            "Closing applications is currently configured for Windows."
        )


    # -----------------------------------------------------
    # CHROME
    # -----------------------------------------------------

    if (
        "chrome" in app_name
        or "google chrome" in app_name
    ):

        return _taskkill(
            "chrome.exe",
            "Chrome"
        )


    # -----------------------------------------------------
    # CALCULATOR
    # -----------------------------------------------------

    if (
        "calculator" in app_name
        or app_name == "calc"
    ):

        return _taskkill(
            "CalculatorApp.exe",
            "Calculator"
        )


    # -----------------------------------------------------
    # NOTEPAD
    # -----------------------------------------------------

    if "notepad" in app_name:

        return _taskkill(
            "notepad.exe",
            "Notepad"
        )


    # -----------------------------------------------------
    # PAINT
    # -----------------------------------------------------

    if "paint" in app_name:

        return _taskkill(
            "mspaint.exe",
            "Paint"
        )


    # -----------------------------------------------------
    # COMMAND PROMPT
    # -----------------------------------------------------

    if (
        "command prompt" in app_name
        or app_name == "cmd"
    ):

        return _taskkill(
            "cmd.exe",
            "Command Prompt"
        )


    # -----------------------------------------------------
    # POWERSHELL
    # -----------------------------------------------------

    if (
        "powershell" in app_name
        or "power shell" in app_name
    ):

        return _taskkill(
            "powershell.exe",
            "PowerShell"
        )


    return action_result(
        False,
        "close_app",
        f"I don't have permission to close '{app_name}'. "
        "That application is not in my safe action list."
    )


# =========================================================
# TASKKILL
# =========================================================

def _taskkill(
    process_name: str,
    display_name: str
):

    try:

        result = subprocess.run(

            [
                "taskkill",
                "/F",
                "/IM",
                process_name,
            ],

            capture_output=True,

            text=True,

            timeout=5,

        )


        if result.returncode == 0:

            return action_result(
                True,
                "close_app",
                f"{display_name} closed."
            )


        output = (
            (result.stdout or "")
            + " "
            + (result.stderr or "")
        ).lower()


        if (
            "not found" in output
            or
            "no running instance" in output
        ):

            return action_result(
                True,
                "close_app",
                f"{display_name} is not currently running."
            )


        return action_result(
            False,
            "close_app",
            f"I couldn't close {display_name}.",
            result.stderr.strip()
        )


    except Exception as error:

        return action_result(
            False,
            "close_app",
            f"I couldn't close {display_name}.",
            str(error)
        )


# =========================================================
# OPEN SAFE FOLDER
# =========================================================

def open_folder(
    folder_name: str
):

    folder_name = (
        folder_name
        .strip()
        .lower()
    )

    if platform.system() != "Windows":

        return action_result(
            False,
            "open_folder",
            "Opening folders is currently configured for Windows."
        )


    home = os.path.expanduser(
        "~"
    )


    folders = {

        "desktop":
            os.path.join(
                home,
                "Desktop"
            ),

        "documents":
            os.path.join(
                home,
                "Documents"
            ),

        "downloads":
            os.path.join(
                home,
                "Downloads"
            ),

        "pictures":
            os.path.join(
                home,
                "Pictures"
            ),

        "videos":
            os.path.join(
                home,
                "Videos"
            ),

        "music":
            os.path.join(
                home,
                "Music"
            ),

    }


    path = folders.get(
        folder_name
    )


    if not path:

        return action_result(
            False,
            "open_folder",
            "I can only open safe folders such as "
            "Desktop, Documents, Downloads, Pictures, "
            "Videos and Music."
        )


    if not os.path.exists(
        path
    ):

        return action_result(
            False,
            "open_folder",
            f"The {folder_name} folder does not exist."
        )


    try:

        os.startfile(
            path
        )

        return action_result(
            True,
            "open_folder",
            f"Opening {folder_name}."
        )

    except Exception as error:

        return action_result(
            False,
            "open_folder",
            f"I couldn't open {folder_name}.",
            str(error)
        )


# =========================================================
# ACTION DETECTION
# =========================================================

def detect_action(
    text: str
):

    normalized = normalize_command(
            text
        )


    # =====================================================
    # WEBSITE COMMANDS
    # =====================================================

    website_prefixes = [

        "open website ",

        "open site ",

        "go to ",

        "visit ",

    ]


    for prefix in website_prefixes:

        if normalized.startswith(
            prefix
        ):

            return {

                "action":
                    "open_website",

                "target":
                    extract_target(
                        text,
                        website_prefixes
                    )

            }


    # =====================================================
    # "OPEN YOUTUBE" / "OPEN GOOGLE" ETC.
    # =====================================================

    if normalized.startswith(
        "open "
    ):

        target = (
            normalized[
                len("open "):
            ]
            .strip()
        )


        if target in WEBSITE_ALIASES:

            return {

                "action":
                    "open_website",

                "target":
                    target

            }


    # =====================================================
    # FOLDER COMMANDS
    # =====================================================

    folder_prefixes = [

        "open folder ",

        "open my folder ",

        "open directory ",

    ]


    for prefix in folder_prefixes:

        if normalized.startswith(
            prefix
        ):

            return {

                "action":
                    "open_folder",

                "target":
                    extract_target(
                        text,
                        folder_prefixes
                    )

            }


    # =====================================================
    # OPEN APPLICATION
    # =====================================================

    open_prefixes = [

        "open ",

        "launch ",

        "start ",

        "run ",

    ]


    known_apps = [

        "chrome",

        "google chrome",

        "calculator",

        "calc",

        "notepad",

        "paint",

        "mspaint",

        "file explorer",

        "explorer",

        "command prompt",

        "cmd",

        "powershell",

        "power shell",

    ]


    for prefix in open_prefixes:

        if normalized.startswith(
            prefix
        ):

            target = extract_target(
                text,
                open_prefixes
            )


            target_lower = (
                target.lower()
            )


            if any(
                app in target_lower
                for app in known_apps
            ):

                return {

                    "action":
                        "open_app",

                    "target":
                        target

                }


    # =====================================================
    # CLOSE APPLICATION
    # =====================================================

    close_prefixes = [

        "close ",

        "quit ",

        "exit ",

        "stop ",

    ]


    for prefix in close_prefixes:

        if normalized.startswith(
            prefix
        ):

            target = extract_target(
                text,
                close_prefixes
            )


            target_lower = (
                target.lower()
            )


            if any(
                app in target_lower
                for app in known_apps
            ):

                return {

                    "action":
                        "close_app",

                    "target":
                        target

                }


    return None


# =========================================================
# EXECUTE
# =========================================================

def execute_action(
    action_data
):

    if not action_data:

        return action_result(
            False,
            "none",
            "I couldn't determine an action."
        )


    action = action_data.get(
            "action"
        )


    target = action_data.get(
            "target",
            ""
        )


    if action == "open_app":

        return open_application(
            target
        )


    if action == "close_app":

        return close_application(
            target
        )


    if action == "open_website":

        return open_website(
            target
        )


    if action == "open_folder":

        return open_folder(
            target
        )


    return action_result(
        False,
        action or "unknown",
        "That action is not implemented."
    )


# =========================================================
# PUBLIC ACTION HANDLER
# =========================================================

def handle_action(
    text: str
):

    action_data = (
        detect_action(
            text
        )
    )


    if not action_data:

        return None


    return execute_action(
        action_data
    )
