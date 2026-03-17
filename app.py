# app.py —— Shiny 前端（UI + 调用），推理细节在 bp_infer.py
from shiny import App, ui, reactive, render
import numpy as np
from Model_infer import predict_proba

THRESHOLD = 0.136  # 如需对齐论文Specificity≈95%，可用阈值扫描后再改

# 20个测量条目；计算时转换为 8 个特征
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h5("请输入 20 个测量条目："),

        # 第一组：Exposure_DrugMisuser（条目 1）
        ui.h6("Exposure_DrugMisuser"),
        ui.p("您周围有吸食毒品或者新型精神活性物质（NPS）的人吗？"),
        ui.input_select("item1", "请选择：", {"0": "没有", "1": "有"}),
        ui.hr(),

        # 第二组：WorkNightlife（条目 2）
        ui.h6("WorkNightlife"),
        ui.p("请评估自己是否经常处在夜生活、娱乐场所或相关工作环境中，这些场所可能更容易接触到药物。"),
        ui.input_checkbox_group("item2","请选择所有符合的选项：",{
                                                                "1": "酒吧",
                                                                "2": "网吧",
                                                                "3": "舞厅",
                                                                "4": "夜总会",
                                                                "5": "KTV",
                                                                "6": "迪厅",
                                                                "7": "休闲会所",
                                                                "8": "农家乐",
                                                                "0": "以上均不是"}),
        ui.hr(),

        # 第三组：New_scale_ACEs（条目 3–7）
        ui.h6("ACEs"),
        ui.p("对于以下描述的情境，在18岁以前，你是否有经历过？请选择所有符合的选项："),
        ui.input_select("item3", "心理虐待，例如父母或其他大人常常或不时对你恶言相向，羞辱、咒骂、贬低、打击你？让你担心被伤害？",
        {"0": "都没经历过", "1": "是，曾经历过其中1种", "2": "是，曾经历过两种及以上"}),
        ui.input_select("item4", "生理虐待，例如父母或其他大人常常或经常动粗，推你、用力抓人、打耳光、对你丢东西？或是把你揍得黑青、受伤？",
        {"0": "都没经历过", "1": "是，曾经历过其中1种", "2": "是，曾经历过两种及以上"}),
        ui.input_select("item5", "性虐待，例如有没有大人或年长你5岁以上的人，曾经乱摸你、调戏抚弄、或要你摸他们的身体？或想要让你跟他们性交（口交、肛交或生殖器接触）？",
        {"0": "都没经历过", "1": "是，曾经历过其中1种", "2": "是，曾经历过两种及以上"}),
        ui.input_select("item6", "心理上或生理上的需求忽视，例如觉得家里没人爱我、认为我不重要、家人之间不亲近、不会彼此照顾或支持，或觉得吃不饱、要穿脏衣服、没有人会保护我，或父母总是喝醉酒、吸毒脑袋不清楚，所以没办法照顾我或带我去看医生。",
        {"0": "都没经历过", "1": "是，曾经历过其中1种", "2": "是，曾经历过两种及以上"}),
        ui.input_select("item7", "家庭功能障碍，包括父母离婚，目睹家暴，如常常见到母亲或继母被推搡、打耳光、被东西砸或被用刀或类似物品威胁，或者家庭中有人患有精神疾病、吸毒、犯罪等。",
        {"0": "都没经历过", "1": "是，曾经历过其中1种", "2": "是，曾经历过两种及以上"}),
        ui.hr(),

        # 第四组：TradDrug_Cog（条目 8）
        ui.h6("TradDrug_Cog"),
        ui.p("您知道的传统毒品种类有哪些？"),
        ui.input_checkbox_group("item8","请选择所有符合的选项：",{
                                                                "1": "白粉（海洛因/二乙酰吗啡）",
                                                                "2": "大烟/鸦片",
                                                                "3": "冰毒（甲基安非他明/甲基苯丙胺）",
                                                                "4": "K粉（氯胺酮）",
                                                                "5": "摇头丸（亚甲二氧基甲基安非他明）",
                                                                "6": "大麻/山丝苗/火麻",
                                                                "7": "可卡因/古柯碱（苯甲基芽子碱）",
                                                                "8": "麻古（主要成分是冰毒）",
                                                                "0": "无"}),
        ui.hr(),

        # 第五组：New_scale_RSEDM（条目 9–12）
        ui.h6("RSEDM"),
        ui.p("在以下描述的情境中，您有多少信心可以拒绝使用毒品？"),
        ui.input_select("item9", "当同伴或朋友邀请我使用毒品时。",
        {"1": "非常容易拒绝", "2": "容易拒绝", "3": "不确定", "4": "有点难拒绝", "5": "非常难拒绝"}),
        ui.input_select("item10", "当处在可以获得毒品的环境/情境时。",
        {"1": "非常容易拒绝", "2": "容易拒绝", "3": "不确定", "4": "有点难拒绝", "5": "非常难拒绝"}),
        ui.input_select("item11", "当心情不好时（包括但不限于感到悲伤、生气、沮丧、无助、孤独或无聊等负面情绪）。",
        {"1": "非常容易拒绝", "2": "容易拒绝", "3": "不确定", "4": "有点难拒绝", "5": "非常难拒绝"}),
        ui.input_select("item12", "当感觉心情好时（包括但不限于开心、兴奋、放松等正面情绪）。",
        {"1": "非常容易拒绝", "2": "容易拒绝", "3": "不确定", "4": "有点难拒绝", "5": "非常难拒绝"}),
        ui.hr(),

        # 第六组：New_scale_SS_3（条目 13–15）
        ui.h6("SS"),
        ui.p("您有多大程度同意以下是针对您人格特质的陈述？"),
        ui.input_select("item13", "只要是别人没有做过或我未曾尝试过的事，我都非常想试一试。",
        {"1": "完全不同意", "2": "基本不同意", "3": "中立", "4": "基本同意", "5": "完全同意"}),
        ui.input_select("item14", "我对新事物和冒险尤为感兴趣，特别是冒险总是让我开心，为了追求新的刺激和刺激，我有时会不顾或违反既有规则。",
        {"1": "完全不同意", "2": "基本不同意", "3": "中立", "4": "基本同意", "5": "完全同意"}),
        ui.input_select("item15", "如果长期做重复的事情或在同一个地方待太久，我会感到烦躁或不安。",
        {"1": "完全不同意", "2": "基本不同意", "3": "中立", "4": "基本同意", "5": "完全同意"}),
        ui.hr(),

        # 第七组：New_scale_EPB（条目 16–17）
        ui.h6("EPB"),
        ui.p("您有多大程度同意以下是针对您日常行为的陈述？"),
        ui.input_select("item16", "我总是不遵守规则，例如撒谎、离家出走、没有父母允许，擅自饮酒、抽烟、使用烟草产品、电子烟甚至吸毒，违反校规校纪、逃学、旷课、放火等，做这些行为后我不感到内疚。",
        {"0": "不符合", "1": "有点符合", "2": "非常符合"}),
        ui.input_select("item17", "我时常有攻击行为，例如与他人争论、毁坏东西、打架、尖叫、戏弄、恐吓、殴打他人、待人苛刻、多疑、脾气暴躁。",
        {"0": "不符合", "1": "有点符合", "2": "非常符合"}),
        ui.hr(),

        # 第八组：New_scale_PADM（条目 18–20）
        ui.h6("PADM"),
        ui.p("您有多大程度同意以下观点？"),
        ui.input_select("item18", "有些人认为吸毒有益处，例如可以抛开烦恼、缓解紧张情绪、令人开心、可以结交朋友，我认同这种看法。",
        {"1": "非常不同意", "2": "不同意", "3": "很难说", "4": "同意", "5": "非常同意"}),
        ui.input_select("item19", "我认为毒品的危害不大且主要是暂时的，只要不上瘾就没有永久伤害，而且我有信心有能力控制自己不上瘾。",
        {"1": "非常不同意", "2": "不同意", "3": "很难说", "4": "同意", "5": "非常同意"}),
        ui.input_select("item20", "吸毒是青少年群体的一种喜好，尝试吸毒并不是什么大不了的事。",
        {"1": "非常不同意", "2": "不同意", "3": "很难说", "4": "同意", "5": "非常同意"}),
        ui.hr(),

        ui.input_action_button("calc", "计算", class_="btn-primary"),
        ui.hr(),
        ui.p("结果仅供科研参考。"), 
        width="1200px"
    ),
    ui.panel_title("NPS使用风险评估计算器"),
    ui.card(ui.h4("结果"), ui.output_text("label"), ui.output_text("prob"), class_="mt-3")
)

def server(input, output, session):
    result = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.calc)
    def _do_predict():
        # 1. Exposure_DrugMisuser：直接对应二分类变量
        Exposure_DrugMisuser = int(input.item1())

        # 2. WorkNightlife：只要选了1-8任一个，就记1；选0或没选记0
        item2_selected = input.item2() or []
        WorkNightlife = 0 if ("0" in item2_selected or len(item2_selected) == 0) else 1

        # 3. ACEs：item3-item7加和，若和 >= 2 则记1，否则记0
        ace_sum = (
            int(input.item3()) +
            int(input.item4()) +
            int(input.item5()) +
            int(input.item6()) +
            int(input.item7())
        )
        New_scale_ACEs = 1 if ace_sum >= 2 else 0

        # 4. TradDrug_Cog：只记知道几种；若选“无”则记0
        item8_selected = input.item8() or []
        TradDrug_Cog = 0 if ("0" in item8_selected) else len(item8_selected)

        # 5. RSEDM：item9-item12求和
        New_scale_RSEDM = (
            int(input.item9()) +
            int(input.item10()) +
            int(input.item11()) +
            int(input.item12())
        )

        # 6. SS：item13-item15求和
        New_scale_SS_3 = (
            int(input.item13()) +
            int(input.item14()) +
            int(input.item15())
        )

        # 7. EPB：item16-item17求和
        New_scale_EPB = (
            int(input.item16()) +
            int(input.item17())
        )

        # 8. PADM：item18-item20求和
        New_scale_PADM = (
            int(input.item18()) +
            int(input.item19()) +
            int(input.item20())
        )
        
        # 按模型需要的8个特征顺序组成输入
        x8 = np.array([[
            Exposure_DrugMisuser,
            WorkNightlife,
            New_scale_ACEs,
            TradDrug_Cog,
            New_scale_RSEDM,
            New_scale_SS_3,
            New_scale_EPB,
            New_scale_PADM
        ]], dtype=np.float64)


        prob = predict_proba(x8)
        label = "高风险" if prob >= THRESHOLD else "低风险"
        result.set((label, prob))

    @output 
    @render.text
    def label():
        r = result.get()
        return "点击“计算”查看判定。" if r is None else f"判定：{r[0]}"

    @output 
    @render.text
    def prob():
        r = result.get()
        return "" if r is None else f"概率：{r[1]:.4f}（阈值 {THRESHOLD:.2f}）"

app = App(app_ui, server)

#启动命令：复制到下面的终端运行
# python -m shiny run --reload app.py

