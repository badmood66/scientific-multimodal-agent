from pathlib import Path

from scimultimodal_agent.data.chartqa import download_chartqa

if __name__ == "__main__":
    print(download_chartqa("test", Path("D:/ResearchData/hf_cache/hub")))

