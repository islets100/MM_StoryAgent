import time
import json
from pathlib import Path

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from .base import init_tool_instance


class MMStoryAgent:

    def __init__(self) -> None:
        # 临时禁用 speech，避免需要阿里云 API 凭证
        # 如需启用语音，请先设置环境变量：.\setup_env.ps1
        self.modalities = ["image"]  # 原来是 ["image", "speech"]

    def call_modality_agent(self, modality, agent, params, return_dict):
        result = agent.call(params)
        return_dict[modality] = result

    def write_story(self, config):
        cfg = config["story_writer"]
        story_writer = init_tool_instance(cfg)
        pages = story_writer.call(cfg["params"])
        return pages
    
    def generate_modality_assets(self, config, pages):
        script_data = {"pages": [{"story": page} for page in pages]}
        story_dir = Path(config["story_dir"])

        for sub_dir in self.modalities:
            (story_dir / sub_dir).mkdir(exist_ok=True, parents=True)

        agents = {}
        params = {}
        for modality in self.modalities:
            agents[modality] = init_tool_instance(config[modality + "_generation"])
            params[modality] = config[modality + "_generation"]["params"].copy()
            params[modality].update({
                "pages": pages,
                "save_path": story_dir / modality
            })

        processes = []
        return_dict = mp.Manager().dict()

        for modality in self.modalities:
            p = mp.Process(
                target=self.call_modality_agent,
                args=(
                    modality,
                    agents[modality],
                    params[modality],
                    return_dict)
                )
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()

        for modality, result in return_dict.items():
            try:
                if modality == "image":
                    images = result["generation_results"]
                    for idx in range(len(pages)):
                        script_data["pages"][idx]["image_prompt"] = result["prompts"][idx]
                elif modality == "sound":
                    for idx in range(len(pages)):
                        script_data["pages"][idx]["sound_prompt"] = result["prompts"][idx]
                elif modality == "music":
                    script_data["music_prompt"] = result["prompt"]
            except Exception as e:
                print(f"Error occurred during generation: {e}")
        
        with open(story_dir / "script_data.json", "w") as writer:
            json.dump(script_data, writer, ensure_ascii=False, indent=4)
        
        return images
    
    def compose_storytelling_video(self, config, pages):
        video_compose_agent = init_tool_instance(config["video_compose"])
        params = config["video_compose"]["params"].copy()
        params["pages"] = pages
        video_compose_agent.call(params)

    def call(self, config):
        pages = self.write_story(config)
        images = self.generate_modality_assets(config, pages)
        # 启用视频合成
        self.compose_storytelling_video(config, pages)
        print("\n" + "="*60)
        print("✅ 故事生成、图像生成和视频合成完成！")
        print("📁 输出目录:", config.get("story_dir", "generated_stories/example"))
        print("="*60)
