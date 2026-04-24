import os
import sys
from rag_system import RAGSystem
from rag_system.core import RAGConfig, RAGQuery

if __name__ == '__main__':
    # Accept query from command line argument or use default
    if len(sys.argv) > 1:
        query_text = ' '.join(sys.argv[1:])
    else:
        query_text = "What is strait of Hormouz?"
    
    sys.stdout.write(f"Running custom query: {query_text}\n\n")

    config = RAGConfig()
    rag = RAGSystem(
        config,
        document_path='state_of_the_union.txt',
        llm_api_token=os.getenv('REPLICATE_API_TOKEN')
    )

    response = rag.query(RAGQuery(query_text=query_text, top_k=5))

    output = []
    output.append('\n--- RESPONSE ---')
    output.append('Answer:')
    output.append(response.answer)
    output.append(f'Latency: {response.latency_ms:.2f}ms')
    output.append(f'Retrieved chunks: {len(response.retrieved_chunks)}')
    output.append('\nCHUNKS:')
    for chunk in response.retrieved_chunks:
        excerpt = chunk.chunk.content[:160].replace('\n', ' ')
        output.append(f'Rank {chunk.rank}, score={chunk.score:.4f}, excerpt={excerpt}')

    sys.stdout.buffer.write(('\n'.join(output) + '\n').encode('utf-8', 'replace'))
