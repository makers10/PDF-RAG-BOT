# benchmark_comparison.py
"""
Compare your RAG system against baseline metrics
"""
import time
import pickle
from rag_pipeline import answer_question

# Load vector store
vector_store_file = "Solar-System-GK-Notes-in-PDF.pdf.pkl"
with open(vector_store_file, "rb") as f:
    vector_store = pickle.load(f)

# Benchmark questions (various difficulty levels)
benchmark_questions = {
    "Easy": [
        "What is the sun?",
        "How many planets are there?",
        "What is Earth?"
    ],
    "Medium": [
        "What is the difference between inner and outer planets?",
        "Why is Mars called the red planet?",
        "What makes Earth suitable for life?"
    ],
    "Hard": [
        "How does the sun provide energy to Earth?",
        "What is the relationship between planets and their moons?",
        "Why do planets orbit the sun?"
    ]
}

print("=" * 80)
print("RAG SYSTEM BENCHMARK - DIFFICULTY LEVELS")
print("=" * 80)

overall_results = {}

for difficulty, questions in benchmark_questions.items():
    print(f"\n{'='*80}")
    print(f"{difficulty.upper()} QUESTIONS")
    print(f"{'='*80}")
    
    times = []
    answers = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{difficulty} Q{i}] {question}")
        
        start = time.time()
        answer = answer_question(
            vector_store,
            question,
            top_k=4,
            similarity_threshold=1.5,
            use_hybrid=True,
            return_sources=False
        )
        elapsed = time.time() - start
        
        times.append(elapsed)
        answers.append(answer)
        
        print(f"Answer: {answer[:100]}..." if len(answer) > 100 else f"Answer: {answer}")
        print(f"Time: {elapsed:.2f}s")
        
        # Quality check
        if len(answer) > 20 and "couldn't" not in answer.lower():
            print("Quality: ✅ Good")
        else:
            print("Quality: ⚠️ Needs review")
    
    avg_time = sum(times) / len(times)
    overall_results[difficulty] = {
        "avg_time": avg_time,
        "questions": len(questions),
        "answers": answers
    }
    
    print(f"\n{difficulty} Average Time: {avg_time:.2f}s")

# Overall summary
print("\n" + "=" * 80)
print("OVERALL BENCHMARK SUMMARY")
print("=" * 80)

total_questions = sum(r['questions'] for r in overall_results.values())
total_avg_time = sum(r['avg_time'] for r in overall_results.values()) / len(overall_results)

print(f"Total Questions Tested: {total_questions}")
print(f"Overall Average Response Time: {total_avg_time:.2f}s")

print("\nPerformance by Difficulty:")
for difficulty, results in overall_results.items():
    print(f"  {difficulty}: {results['avg_time']:.2f}s")

# Industry benchmarks
print("\n" + "=" * 80)
print("COMPARISON WITH INDUSTRY STANDARDS")
print("=" * 80)
print("Industry Benchmarks:")
print("  ⚡ Fast: < 2s")
print("  ✅ Good: 2-5s")
print("  ⚠️  Acceptable: 5-10s")
print("  ❌ Slow: > 10s")

if total_avg_time < 2:
    rating = "⚡ FAST - Exceeds industry standards!"
elif total_avg_time < 5:
    rating = "✅ GOOD - Meets industry standards"
elif total_avg_time < 10:
    rating = "⚠️  ACCEPTABLE - Room for improvement"
else:
    rating = "❌ SLOW - Needs optimization"

print(f"\nYour System: {total_avg_time:.2f}s - {rating}")
print("=" * 80)
