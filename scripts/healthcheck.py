import os, platform
print("Python:", platform.python_version())
print("JAVA_HOME:", os.environ.get("JAVA_HOME"))

try:
    import faiss
    print("FAISS:", faiss.__version__)
except Exception as e:
    print("FAISS import failed:", e)

try:
    import pyserini
    print("Pyserini:", pyserini.__version__)
    # light JVM poke
    from pyserini.search import SimpleSearcher  # noqa
    print("Pyserini JVM bridge OK")
except Exception as e:
    print("Pyserini import failed:", e)
