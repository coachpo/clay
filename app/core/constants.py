class Constants:
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    ROLE_TOOL = "tool"
    ROLE_USER = "user"

    CONTENT_DOCUMENT = "document"
    CONTENT_IMAGE = "image"
    CONTENT_TEXT = "text"
    CONTENT_THINKING = "thinking"
    CONTENT_TOOL_RESULT = "tool_result"
    CONTENT_TOOL_USE = "tool_use"

    DELTA_INPUT_JSON = "input_json_delta"
    DELTA_TEXT = "text_delta"

    EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
    EVENT_CONTENT_BLOCK_START = "content_block_start"
    EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
    EVENT_ERROR = "error"
    EVENT_MESSAGE_DELTA = "message_delta"
    EVENT_MESSAGE_START = "message_start"
    EVENT_MESSAGE_STOP = "message_stop"
    EVENT_PING = "ping"

    STOP_END_TURN = "end_turn"
    STOP_MAX_TOKENS = "max_tokens"
    STOP_REFUSAL = "refusal"
    STOP_STOP_SEQUENCE = "stop_sequence"
    STOP_TOOL_USE = "tool_use"

    TOOL_FUNCTION = "function"
