from memory.vector_store import search_memory

def retrieve_context(query):
    memories = search_memory(query)

    if not memories:
        return ""

    return "\n".join(memories)