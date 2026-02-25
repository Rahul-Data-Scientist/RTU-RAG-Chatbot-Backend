from datetime import datetime


def retrieve_all_threads(checkpointer):
    threads_dict = {}

    for checkpoint in checkpointer.list(None):

        config = checkpoint.config.get("configurable", {})
        thread_id = config.get("thread_id")

        if not thread_id:
            continue

        # --- timestamp ---
        ts = checkpoint.checkpoint["ts"]
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

        # --- state values ---
        channel_values = checkpoint.checkpoint.get("channel_values", {})
        title = channel_values.get("conversation_title")

        if thread_id not in threads_dict:
            threads_dict[thread_id] = {
                "last_active": ts,
                "title": title,
            }
        else:
            threads_dict[thread_id]["last_active"] = max(
                threads_dict[thread_id]["last_active"], ts
            )

            # update title if generated later
            if (
                title
                and not threads_dict[thread_id]["title"]
            ):
                threads_dict[thread_id]["title"] = title

    # --- sort by last activity ---
    sorted_threads = sorted(
        threads_dict.items(),
        key=lambda x: x[1]["last_active"],
        reverse=True,
    )

    # --- return clean API response ---
    return [
        {
            "thread_id": thread_id,
            "title": data["title"] or "New Conversation",
            "last_active": data["last_active"].isoformat(),
        }
        for thread_id, data in sorted_threads
    ]


def delete_thread(checkpointer, thread_id: str):
    """
    Permanently delete all checkpoints for a thread_id
    """

    conn = checkpointer.conn

    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM checkpoints
            WHERE thread_id = %s
            """,
            (thread_id,),
        )

    conn.commit()

    return {
        "status": "deleted",
        "thread_id": thread_id,
    }

def rename_thread(chatbot, thread_id: str, new_title: str):

    chatbot.update_state(
        config={"configurable": {"thread_id": thread_id}},
        values={
            "conversation_title": new_title
        },
    )

    return {"status": "ok"}
