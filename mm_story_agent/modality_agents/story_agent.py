import json
from typing import Dict
import random

from tqdm import trange, tqdm

from ..utils.llm_output_check import parse_list
from ..base import register_tool, init_tool_instance
from ..prompts_en import question_asker_system, expert_system, \
    dlg_based_writer_system, dlg_based_writer_prompt, chapter_writer_system, \
    data_based_writer_system, data_based_writer_prompt, \
    long_text_parser_system, long_text_parser_prompt


def json_parse_outline(outline):
    outline = outline.strip("```json").strip("```")
    try:
        outline = json.loads(outline)
        if not isinstance(outline, dict):
            return False
        if outline.keys() != {"story_title", "story_outline"}:
            return False
        for chapter in outline["story_outline"]:
            if chapter.keys() != {"chapter_title", "chapter_summary"}:
                return False
    except json.decoder.JSONDecodeError:
        return False
    return True


@register_tool("qa_outline_story_writer")
class QAOutlineStoryWriter:

    def __init__(self,
                 cfg: Dict):
        self.cfg = cfg
        self.temperature = cfg.get("temperature", 1.0)
        self.max_conv_turns = cfg.get("max_conv_turns", 3)
        self.num_outline = cfg.get("num_outline", 4)
        self.llm_type = cfg.get("llm", "qwen")

    def generate_outline(self, params):
        # `params`: story setting like 
        # {
        #     "story_title": "xxx",
        #     "main_role": "xxx",
        #     ......
        # }
        asker = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": question_asker_system,
                "track_history": False
            }
        })
        expert = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": expert_system,
                "track_history": False
            }
        })

        dialogue = []
        for turn in trange(self.max_conv_turns):
            dialogue_history = "\n".join(dialogue)
            
            question, success = asker.call(
                f"Story setting: {params}\nDialogue history: \n{dialogue_history}\n",
                temperature=self.temperature
            )
            question = question.strip()
            if question == "Thank you for your help!":
                break
            dialogue.append(f"You: {question}")
            answer, success = expert.call(
                f"Story setting: {params}\nQuestion: \n{question}\nAnswer: ",
                temperature=self.temperature
            )
            answer = answer.strip()
            dialogue.append(f"Expert: {answer}")

        # print("\n".join(dialogue))
        writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": dlg_based_writer_system,
                "track_history": False
            }
        })
        writer_prompt = dlg_based_writer_prompt.format(
            story_setting=params,
            dialogue_history="\n".join(dialogue),
            num_outline=self.num_outline
        )

        outline, success = writer.call(writer_prompt, success_check_fn=json_parse_outline)
        outline = json.loads(outline)
        # print(outline)
        return outline

    def generate_story_from_outline(self, outline):
        chapter_writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": chapter_writer_system,
                "track_history": False
            }
        })
        all_pages = []
        for idx, chapter in enumerate(tqdm(outline["story_outline"])):
            chapter_detail, success = chapter_writer.call(
                json.dumps(
                    {
                        "completed_story": all_pages,
                        "current_chapter": chapter
                    },
                    ensure_ascii=False
                ),
                success_check_fn=parse_list,
                temperature=self.temperature
            )
            while success is False:
                chapter_detail, success = chapter_writer.call(
                    json.dumps(
                        {
                            "completed_story": all_pages,
                            "current_chapter": chapter
                        },
                        ensure_ascii=False
                    ),
                    seed=random.randint(0, 100000),
                    temperature=self.temperature,
                    success_check_fn=parse_list
                )
            pages = [page.strip() for page in eval(chapter_detail)]
            all_pages.extend(pages)
        # print(all_pages)
        return all_pages

    def call(self, params):
        print("\n📖 使用 QA-Outline 模式生成故事")
        print("   模式: Story Topic -> Outline -> Story Pages")
        print(f"   输入参数: {params}")
        
        print("\n📝 步骤 1/2: 生成故事大纲...")
        outline = self.generate_outline(params)
        print(f"   ✅ 大纲生成完成: {outline['story_title']}")
        print(f"   章节数: {len(outline['story_outline'])}")
        
        print("\n📝 步骤 2/2: 根据大纲生成故事内容...")
        pages = self.generate_story_from_outline(outline)
        print(f"   ✅ 故事生成完成，共 {len(pages)} 页")
        
        # 调试：打印生成的故事页面
        print("\n🔍 生成的故事内容预览:")
        for idx, page in enumerate(pages[:3]):  # 只显示前3页
            print(f"   [第 {idx + 1} 页] {page[:80]}...")
        if len(pages) > 3:
            print(f"   ... 还有 {len(pages) - 3} 页")
        
        return pages


@register_tool("data_based_story_writer")
class DataBasedStoryWriter:
    """
    数据驱动的故事生成器
    直接从结构化数据或长文本生成故事，跳过大纲生成步骤
    架构: Data/Long Text -> Story Pages
    """
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.temperature = cfg.get("temperature", 1.0)
        self.llm_type = cfg.get("llm", "qwen")
        self.max_retries = cfg.get("max_retries", 3)
    
    def parse_long_text(self, long_text: str) -> Dict:
        """
        将长文本解析为结构化数据
        
        Args:
            long_text: 长文本描述
            
        Returns:
            Dict: 结构化的故事数据
        """
        print("\n🔄 检测到长文本输入，正在解析...")
        print(f"📝 长文本内容:\n{long_text[:200]}...\n")
        
        # 初始化解析器
        parser = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": long_text_parser_system,
                "track_history": False
            }
        })
        
        # 格式化 prompt
        parser_prompt = long_text_parser_prompt.format(long_text=long_text)
        
        # 解析长文本
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                parsed_data, success = parser.call(
                    parser_prompt,
                    temperature=self.temperature
                )
                
                # 尝试解析 JSON
                parsed_data = parsed_data.strip("```json").strip("```").strip()
                structured_data = json.loads(parsed_data)
                
                print("✅ 长文本解析成功！")
                print(f"🔍 解析结果:\n{json.dumps(structured_data, ensure_ascii=False, indent=2)}\n")
                
                return structured_data
                
            except json.JSONDecodeError as e:
                retry_count += 1
                print(f"   ⚠️ JSON 解析失败: {str(e)}")
                print(f"   ⚠️ 重试 {retry_count}/{self.max_retries}...")
            except Exception as e:
                retry_count += 1
                print(f"   ❌ 错误: {str(e)}")
                print(f"   ⚠️ 重试 {retry_count}/{self.max_retries}...")
        
        raise RuntimeError(f"长文本解析失败，已重试 {self.max_retries} 次")
    
    def prepare_story_data(self, params: Dict) -> Dict:
        """
        准备故事数据，支持长文本和结构化数据两种输入
        
        Args:
            params: 输入参数
            
        Returns:
            Dict: 标准化的故事数据
        """
        # 检查是否包含长文本
        if "long_text" in params and params["long_text"]:
            long_text = params["long_text"].strip()
            if long_text:
                # 解析长文本
                structured_data = self.parse_long_text(long_text)
                
                # 合并其他参数（如果有的话）
                for key in ["num_pages", "theme", "setting"]:
                    if key in params and params[key]:
                        if key not in structured_data or not structured_data[key]:
                            structured_data[key] = params[key]
                
                return structured_data
        
        # 否则使用结构化数据
        return params
    
    def call(self, params):
        """
        从数据或长文本直接生成故事
        
        Args:
            params: 输入参数，支持两种格式:
            1. 结构化数据:
            {
                "characters": [{"name": "xxx", "description": "xxx"}, ...],
                "setting": "xxx",
                "plot_points": ["xxx", "xxx", ...],
                "theme": "xxx",
                "num_pages": 4
            }
            2. 长文本:
            {
                "long_text": "完整的故事描述...",
                "num_pages": 4  # 可选
            }
        
        Returns:
            List[str]: 故事页面列表
        """
        print("\n📖 使用 Data-Based 模式生成故事")
        print("   模式: Data/Long Text -> Story Pages (跳过大纲)")
        
        # 准备故事数据
        story_data = self.prepare_story_data(params)
        
        print(f"\n📊 最终故事数据:")
        print(f"{json.dumps(story_data, ensure_ascii=False, indent=2)}")
        
        # 初始化 LLM
        writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": data_based_writer_system,
                "track_history": False
            }
        })
        
        # 格式化 prompt
        writer_prompt = data_based_writer_prompt.format(
            story_data=json.dumps(story_data, ensure_ascii=False, indent=2)
        )
        
        print("\n📝 正在生成故事...")
        print(f"🔍 使用的 Prompt:\n{writer_prompt}\n")
        
        # 生成故事
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                story_pages, success = writer.call(
                    writer_prompt,
                    success_check_fn=parse_list,
                    temperature=self.temperature
                )
                
                if success:
                    # 解析结果
                    pages = eval(story_pages)
                    pages = [page.strip() for page in pages]
                    
                    print(f"   ✅ 故事生成完成，共 {len(pages)} 页")
                    
                    # 调试：打印生成的故事页面
                    print("\n🔍 生成的故事内容预览:")
                    for idx, page in enumerate(pages):
                        print(f"   [第 {idx + 1} 页] {page}")
                    
                    return pages
                else:
                    retry_count += 1
                    print(f"   ⚠️ 生成失败，重试 {retry_count}/{self.max_retries}...")
                    
            except Exception as e:
                retry_count += 1
                print(f"   ❌ 错误: {str(e)}")
                print(f"   ⚠️ 重试 {retry_count}/{self.max_retries}...")
        
        raise RuntimeError(f"故事生成失败，已重试 {self.max_retries} 次")


@register_tool("unified_story_writer")
class UnifiedStoryWriter:
    """
    统一的故事生成接口
    支持两种模式自由切换：
    1. QA-Outline 模式: Story Topic -> Outline -> Story Pages
    2. Data-Based 模式: Data -> Story Pages
    """
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.mode = cfg.get("mode", "qa_outline")  # "qa_outline" 或 "data_based"
        
        # 根据模式初始化对应的生成器
        if self.mode == "qa_outline":
            self.generator = QAOutlineStoryWriter(cfg)
        elif self.mode == "data_based":
            self.generator = DataBasedStoryWriter(cfg)
        else:
            raise ValueError(f"不支持的模式: {self.mode}，请选择 'qa_outline' 或 'data_based'")
    
    def call(self, params):
        """
        统一的调用接口
        
        Args:
            params: 输入参数
                - QA-Outline 模式: {"story_topic": "xxx", "main_role": "xxx", ...}
                - Data-Based 模式: {"characters": [...], "plot_points": [...], ...}
        
        Returns:
            List[str]: 故事页面列表
        """
        print("\n" + "="*60)
        print(f"📚 统一故事生成器")
        print(f"   当前模式: {self.mode.upper()}")
        print("="*60)
        
        # 调用对应的生成器
        pages = self.generator.call(params)
        
        print("\n" + "="*60)
        print(f"✅ 故事生成完成！")
        print(f"   模式: {self.mode.upper()}")
        print(f"   总页数: {len(pages)}")
        print("="*60 + "\n")
        
        return pages
