"""术语词典构建脚本."""

import json
import re
from collections import Counter
from pathlib import Path

import jieba
import pypinyin


# 预定义的领域核心术语（基于铁路调度场景）
CORE_TERMS = {
    "道岔": {
        "category": "站场设备",
        "aliases": ["道差", "到岔", "到差", "倒岔"],
        "patterns": [r"\d+号道岔", r"道岔定位", r"道岔反位", r"故障道岔"],
    },
    "信号机": {
        "category": "信号设备",
        "aliases": ["新号机", "信号机", "信后机"],
        "patterns": [r"(上行|下行|进站|出站|调车|引导|通过)信号机", r"信号机灯光熄灭"],
    },
    "轨道电路": {
        "category": "信号设备",
        "aliases": ["轨道电路", "轨道线", "轨道回路"],
        "patterns": [r"轨道电路红光带", r"轨道电路故障"],
    },
    "红光带": {
        "category": "信号状态",
        "aliases": ["红光带", "洪光带", "红中带", "红光代"],
        "patterns": [],
    },
    "进路": {
        "category": "行车组织",
        "aliases": ["近路", "进路"],
        "patterns": [r"(接车|发车|调车)进路", r"进路准备好了?", r"准备进路"],
    },
    "发车": {
        "category": "行车组织",
        "aliases": ["发车", "法车", "发扯"],
        "patterns": [r"\d+道发车", r"准备发车"],
    },
    "接车": {
        "category": "行车组织",
        "aliases": ["接车", "结车", "截车"],
        "patterns": [r"\d+道接车", r"引导接车"],
    },
    "闭塞": {
        "category": "行车组织",
        "aliases": ["闭塞", "必色", "闭色"],
        "patterns": [r"闭塞分区", r"区间闭塞", r"办理闭塞"],
    },
    "联锁": {
        "category": "信号",
        "aliases": ["联锁", "连锁", "连索"],
        "patterns": [r"联锁实验", r"联锁故障"],
    },
    "销记": {
        "category": "作业流程",
        "aliases": ["销记", "消记", "小记"],
        "patterns": [r"工务销记", r"电务销记", r"登记.*销记"],
    },
    "绿色许可证": {
        "category": "行车凭证",
        "aliases": ["绿色许可证", "绿色通行证", "绿证"],
        "patterns": [],
    },
    "引导信号": {
        "category": "信号",
        "aliases": ["引导信号", "引道信号", "诱导信号"],
        "patterns": [r"开放引导信号", r"引导信号好了"],
    },
    "扳道员": {
        "category": "人员",
        "aliases": ["扳道员", "扳道元", "搬道员"],
        "patterns": [r"\d+号扳道员"],
    },
    "值班员": {
        "category": "人员",
        "aliases": ["值班员", "值斑员", "直班员"],
        "patterns": [r"(车站|内勤|外勤)值班员", r"报告值班员"],
    },
    "调度员": {
        "category": "人员",
        "aliases": ["调度员", "调渡员", "条度员"],
        "patterns": [r"列车调度员", r"助理调度员"],
    },
    "工务": {
        "category": "单位",
        "aliases": ["工务", "公务", "工无"],
        "patterns": [r"工务(销记|登记|人员|线路)"],
    },
    "电务": {
        "category": "单位",
        "aliases": ["电务", "店务", "电无"],
        "patterns": [r"电务(销记|登记|人员|设备)"],
    },
    "手摇把": {
        "category": "工具",
        "aliases": ["手摇把", "手要把", "手钥把"],
        "patterns": [],
    },
    "钩锁器": {
        "category": "工具",
        "aliases": ["钩锁器", "勾锁器", "钩索器"],
        "patterns": [],
    },
    "转辙机": {
        "category": "设备",
        "aliases": ["转辙机", "转撤机", "转折机"],
        "patterns": [r"转辙机钥匙"],
    },
    "尖轨": {
        "category": "轨道部件",
        "aliases": ["尖轨", "间轨", "尖柜"],
        "patterns": [r"尖轨与基本轨"],
    },
    "基本轨": {
        "category": "轨道部件",
        "aliases": ["基本轨", "鸡本轨", "基本柜"],
        "patterns": [],
    },
    "咽喉区": {
        "category": "站场",
        "aliases": ["咽喉区", "烟后区", "咽喉去", "烟喉区"],
        "patterns": [r"(上行|下行)咽喉区"],
    },
    "闭塞分区": {
        "category": "信号",
        "aliases": ["闭塞分区", "必色分区", "闭色分区"],
        "patterns": [],
    },
    "区间逻辑检查": {
        "category": "信号",
        "aliases": ["区间逻辑检查", "区间逻辑", "区间逻揖检查"],
        "patterns": [r"区间逻辑检查报警"],
    },
    "股道": {
        "category": "站场",
        "aliases": ["股道", "轨道"],
        "patterns": [r"\d+道(空闲|占用)"],
    },
    "调车": {
        "category": "行车组织",
        "aliases": ["调车", "条车", "调扯"],
        "patterns": [r"调车作业", r"调车信号", r"影响进路的调车作业"],
    },
    "限速": {
        "category": "行车命令",
        "aliases": ["限速", "线速", "限苏"],
        "patterns": [r"限速\d+km/h"],
    },
    "封锁": {
        "category": "行车组织",
        "aliases": ["封锁", "峰锁", "封索"],
        "patterns": [r"区间封锁", r"封锁标识"],
    },
    "开通": {
        "category": "状态",
        "aliases": ["开通", "开同", "开通"],
        "patterns": [r"开通.*(区间|进路|道岔)"],
    },
    "加锁": {
        "category": "作业",
        "aliases": ["加锁", "家锁", "加所"],
        "patterns": [r"(定位|反位)加锁", r"加锁好了"],
    },
    "解锁": {
        "category": "作业",
        "aliases": ["解锁", "姐锁", "解所"],
        "patterns": [r"(进路|道岔)解锁", r"解锁好了"],
    },
    "单锁": {
        "category": "作业",
        "aliases": ["单锁", "丹锁", "单所"],
        "patterns": [r"(定位|反位)单锁"],
    },
    "总锁闭": {
        "category": "信号",
        "aliases": ["总锁闭", "总索闭", "总锁币"],
        "patterns": [r"引导总锁闭", r"总锁闭按钮"],
    },
    "计数器": {
        "category": "设备",
        "aliases": ["计数器", "记数器", "技术器"],
        "patterns": [r"计数器由\d+号变为\d+号"],
    },
    "行车日志": {
        "category": "文档",
        "aliases": ["行车日志", "行车日知", "行车日志"],
        "patterns": [],
    },
    "占线簿": {
        "category": "文档",
        "aliases": ["占线簿", "占线薄", "站线簿"],
        "patterns": [r"填写占线簿", r"抹消占线簿"],
    },
    "行车设备检查登记簿": {
        "category": "文档",
        "aliases": ["行车设备检查登记簿", "行车设备检查登记薄"],
        "patterns": [r"登记《行车设备检查登记簿》"],
    },
    "光带": {
        "category": "信号显示",
        "aliases": ["光带", "光代", "光带"],
        "patterns": [r"光带变红", r"信号光带"],
    },
    "表示": {
        "category": "信号状态",
        "aliases": ["表示", "标是", "表是"],
        "patterns": [r"(道岔|控制台)(恢复|无)表示"],
    },
    "密贴": {
        "category": "轨道状态",
        "aliases": ["密贴", "密切", "密帖"],
        "patterns": [r"尖轨与基本轨密贴"],
    },
    "异状": {
        "category": "状态描述",
        "aliases": ["异状", "一状", "异壮"],
        "patterns": [r"无异状无障碍物"],
    },
    "障碍物": {
        "category": "状态描述",
        "aliases": ["障碍物", "账爱物", "障碍无"],
        "patterns": [],
    },
    "机车车辆": {
        "category": "设备",
        "aliases": ["机车车辆", "机车车两", "机车车量"],
        "patterns": [r"无机车车辆占用"],
    },
    "热备": {
        "category": "行车组织",
        "aliases": ["热备", "热被", "热背"],
        "patterns": [r"热备(动车组|内燃机车)"],
    },
    "轨道车": {
        "category": "车辆",
        "aliases": ["轨道车", "轨道扯", "鬼道车"],
        "patterns": [r"轨道车\d+次"],
    },
    "添乘": {
        "category": "作业",
        "aliases": ["添乘", "天乘", "添成"],
        "patterns": [r"添乘巡线", r"等待添乘"],
    },
    "巡线": {
        "category": "作业",
        "aliases": ["巡线", "寻线", "询线"],
        "patterns": [r"出动.*巡线", r"巡线完毕"],
    },
    " fireworks ": {
        "category": "报警",
        "aliases": ["烟火报警", "烟火抱警", "烟火爆警"],
        "patterns": [r"烟火报警"],
    },
    "隧道": {
        "category": "设施",
        "aliases": ["隧道", "遂道", "碎道"],
        "patterns": [r"隧道内"],
    },
    "疏散": {
        "category": "应急",
        "aliases": ["疏散", "蔬散", "疏撒"],
        "patterns": [r"疏散旅客", r"疏散通道"],
    },
    "防灾救援": {
        "category": "应急",
        "aliases": ["防灾救援", "防灾旧援", "防灾求援"],
        "patterns": [r"防灾救援疏散系统"],
    },
    "接触网": {
        "category": "供电",
        "aliases": ["接触网", "接出头", "接触亡", "节触网"],
        "patterns": [r"接触网停电"],
    },
    "停电": {
        "category": "作业",
        "aliases": ["停电", "停点", "亭电"],
        "patterns": [r"接触网停电"],
    },
    "行车限制卡": {
        "category": "文档",
        "aliases": ["行车限制卡", "行车限制咔", "行车现制卡"],
        "patterns": [r"行车限制卡"],
    },
    "改方": {
        "category": "信号操作",
        "aliases": ["改方", "改芳", "改방"],
        "patterns": [r"办理.*改方"],
    },
    "点灯": {
        "category": "信号操作",
        "aliases": ["点灯", "点登", "点灯"],
        "patterns": [r"(出站|进站)信号机.*点灯"],
    },
    "灭灯": {
        "category": "信号状态",
        "aliases": ["灭灯", "蔑灯", "灭登"],
        "patterns": [r"灯光熄灭"],
    },
    "故障通知按钮": {
        "category": "设备",
        "aliases": ["故障通知按钮", "故障通知安钮", "故障通之按钮"],
        "patterns": [r"按下故障通知按钮"],
    },
    "总人解": {
        "category": "信号操作",
        "aliases": ["总人解", "总人姐", "总人介"],
        "patterns": [r"总人解按钮"],
    },
    "控显": {
        "category": "设备",
        "aliases": ["控显", "空显", "控线"],
        "patterns": [r"控显计数器"],
    },
    "HMI屏": {
        "category": "设备",
        "aliases": ["HMI屏", "hmi屏", "HMI平"],
        "patterns": [r"HMI屏显示"],
    },
    "随车机械师": {
        "category": "人员",
        "aliases": ["随车机械师", "随车机戒师", "随车机写师"],
        "patterns": [],
    },
    "列车长": {
        "category": "人员",
        "aliases": ["列车长", "列车掌", "车列长"],
        "patterns": [],
    },
    "乘务员": {
        "category": "人员",
        "aliases": ["乘务员", "乘无员", "乘务元"],
        "patterns": [r"列车乘务员"],
    },
    "调度命令": {
        "category": "文档",
        "aliases": ["调度命令", "调渡命令", "条度命令"],
        "patterns": [],
    },
    "运行计划": {
        "category": "文档",
        "aliases": ["运行计划", "运行记划", "运形计划"],
        "patterns": [r"列车运行计划"],
    },
    "时刻": {
        "category": "时间",
        "aliases": ["时刻", "时客", "时课"],
        "patterns": [r"核对车次、时刻"],
    },
    "命令指示": {
        "category": "文档",
        "aliases": ["命令指示", "命令指是", "命令指事"],
        "patterns": [],
    },
    "机外停车": {
        "category": "行车状态",
        "aliases": ["机外停车", "机外听车", "机外亭车"],
        "patterns": [r"机外停车"],
    },
    "站内设备故障": {
        "category": "设备状态",
        "aliases": ["站内设备故障", "站内设别故障", "站内设备故仗"],
        "patterns": [r"站内设备故障"],
    },
    "停车": {
        "category": "行车状态",
        "aliases": ["停车", "听车", "亭车"],
        "patterns": [r"\d+道停车", r"立即停车"],
    },
    "紧急停车": {
        "category": "行车状态",
        "aliases": ["紧急停车", "紧急听车", "紧集停车"],
        "patterns": [r"呼叫.*紧急停车"],
    },
    "预告": {
        "category": "作业",
        "aliases": ["预告", "欲告", "预告"],
        "patterns": [r"\d+次预告", r"同意.*预告"],
    },
    "报点": {
        "category": "作业",
        "aliases": ["报点", "保点", "报典"],
        "patterns": [r"报点.*\d+次"],
    },
    "过标": {
        "category": "状态",
        "aliases": ["过标", "过彪", "过表"],
        "patterns": [r"尾部过标"],
    },
    "列尾": {
        "category": "车辆部件",
        "aliases": ["列尾", "列伟", "列尾"],
        "patterns": [r"列车尾部标志", r"确认列尾"],
    },
    "尾部标志": {
        "category": "车辆部件",
        "aliases": ["尾部标志", "尾不标志", "尾部标是"],
        "patterns": [],
    },
    "制动": {
        "category": "车辆操作",
        "aliases": ["制动", "致动", "制动"],
        "patterns": [r"紧急制动"],
    },
    "降速": {
        "category": "行车操作",
        "aliases": ["降速", "将速", "降诉"],
        "patterns": [r"降速\d+km/h"],
    },
    "维持运行": {
        "category": "行车状态",
        "aliases": ["维持运行", "维持运形", "唯持运行"],
        "patterns": [r"维持运行"],
    },
    "加强瞭望": {
        "category": "行车要求",
        "aliases": ["加强瞭望", "加强辽望", "加强聊了"],
        "patterns": [r"加强瞭望"],
    },
    "随时停车": {
        "category": "行车要求",
        "aliases": ["随时停车", "随时听车", "随是停车"],
        "patterns": [r"随时停车速度"],
    },
    "出务": {
        "category": "作业",
        "aliases": ["出务", "出无", "出物"],
        "patterns": [r"出务作业"],
    },
    "上道": {
        "category": "作业",
        "aliases": ["上道", "上倒", "上岛"],
        "patterns": [r"上道作业", r"同意上道"],
    },
    "下道": {
        "category": "作业",
        "aliases": ["下道", "下倒", "下岛"],
        "patterns": [r"已下道", r"下道并返回"],
    },
    "立岗": {
        "category": "作业",
        "aliases": ["立岗", "立刚", "立钢"],
        "patterns": [r"立岗接车", r"立岗地点"],
    },
    "交付": {
        "category": "作业",
        "aliases": ["交付", "交付", "交付"],
        "patterns": [r"向司机交付", r"交付.*司机"],
    },
    "核对": {
        "category": "作业",
        "aliases": ["核对", "合对", "核兑"],
        "patterns": [r"核对.*凭证", r"核对无误"],
    },
    "互检": {
        "category": "作业",
        "aliases": ["互检", "互简", "户检"],
        "patterns": [r"互检凭证"],
    },
    "复诵": {
        "category": "作业",
        "aliases": ["复诵", "复送", "付诵"],
        "patterns": [r"复诵后"],
    },
    "指尖": {
        "category": "作业",
        "aliases": ["剑指", "纸间", "指尖"],
        "patterns": [r"剑指控制台"],
    },
    "记录仪": {
        "category": "设备",
        "aliases": ["记录仪", "记绿仪", "记路仪"],
        "patterns": [r"开启记录仪"],
    },
    "防护": {
        "category": "安全",
        "aliases": ["防护", "防无", "防户"],
        "patterns": [r"防护用品", r"防护措施"],
    },
    "走行径路": {
        "category": "安全",
        "aliases": ["走行径路", "走行经路", "走行京路"],
        "patterns": [r"固定走行径路"],
    },
    "安全区域": {
        "category": "安全",
        "aliases": ["安全区域", "安全区或", "安全去域"],
        "patterns": [r"返回安全区域"],
    },
    "防护措施": {
        "category": "安全",
        "aliases": ["防护措施", "防无措施", "防户措施"],
        "patterns": [],
    },
    "企业规定": {
        "category": "规章",
        "aliases": ["企业规定", "企叶规定", "企业规丁"],
        "patterns": [r"企业规定地点"],
    },
    "站细": {
        "category": "规章",
        "aliases": ["站细", "占细", "站系"],
        "patterns": [r"按站细规定"],
    },
    "非正常": {
        "category": "状态",
        "aliases": ["非正常", "非正常", "非正长"],
        "patterns": [r"非正常情况"],
    },
    "行车凭证": {
        "category": "文档",
        "aliases": ["行车凭证", "行车凭正", "行车平证"],
        "patterns": [r"填写行车凭证", r"交递行车凭证"],
    },
    "书面通知": {
        "category": "文档",
        "aliases": ["书面通知", "书棉通知", "书灭通知"],
        "patterns": [r"书面通知"],
    },
    "占用": {
        "category": "状态",
        "aliases": ["占用", "战用", "占用"],
        "patterns": [r"(线路|区段)占用", r"无机车车辆占用"],
    },
    "空闲": {
        "category": "状态",
        "aliases": ["空闲", "控闲", "空显"],
        "patterns": [r"(线路|股道|区段)空闲"],
    },
    "待发": {
        "category": "状态",
        "aliases": ["待发", "待法", "带发"],
        "patterns": [r"\d+道待发"],
    },
    "待发列车": {
        "category": "状态",
        "aliases": ["待发列车", "待法列车", "带发列车"],
        "patterns": [],
    },
    "后续列车": {
        "category": "行车",
        "aliases": ["后续列车", "后续车列", "后序列车"],
        "patterns": [r"放行后续列车", r"扣停后续列车"],
    },
    "相关列车": {
        "category": "行车",
        "aliases": ["相关列车", "相光列车", "相关车列"],
        "patterns": [r"扣停.*相关列车"],
    },
    "限速运行": {
        "category": "命令",
        "aliases": ["限速运行", "线速运行", "限苏运行"],
        "patterns": [r"限速\d+km/h运行"],
    },
    "注意确认": {
        "category": "用语",
        "aliases": ["注意确认", "注义确认", "注意确人"],
        "patterns": [r"注意确认"],
    },
    "加强": {
        "category": "用语",
        "aliases": ["加强", "家强", "加将"],
        "patterns": [r"加强瞭望"],
    },
    "明白": {
        "category": "用语",
        "aliases": ["明白", "名白", "明百"],
        "patterns": [r"(司机|车站).*明白"],
    },
    "收到": {
        "category": "用语",
        "aliases": ["收到", "手到", "搜到"],
        "patterns": [r"收到.*(通知|报警)"],
    },
    "执行": {
        "category": "用语",
        "aliases": ["执行", "直行", "执刑"],
        "patterns": [r"执行.*标"],
    },
    "清楚了": {
        "category": "用语",
        "aliases": ["清楚了", "请楚了", "轻楚了"],
        "patterns": [],
    },
    "按规定": {
        "category": "用语",
        "aliases": ["按规定", "安规定", "按归定"],
        "patterns": [r"按规定办理"],
    },
    "准许": {
        "category": "用语",
        "aliases": ["准许", "准需", "准序"],
        "patterns": [r"准许.*(发车|放行|疏散)"],
    },
    "同意": {
        "category": "用语",
        "aliases": ["同意", "同义", "同易"],
        "patterns": [r"同意.*(发车|预告|上道)"],
    },
    "请求": {
        "category": "用语",
        "aliases": ["请求", "请球", "轻求"],
        "patterns": [r"请求.*(发车|引导|使用)"],
    },
    "报告": {
        "category": "用语",
        "aliases": ["报告", "抱告", "报高"],
        "patterns": [r"报告值班员"],
    },
    "通知": {
        "category": "用语",
        "aliases": ["通知", "通之", "同知"],
        "patterns": [r"通知.*(司机|工务|电务)"],
    },
    "请立即": {
        "category": "用语",
        "aliases": ["请立即", "请及即", "请既即"],
        "patterns": [r"请立即.*(处理|抢修|上岗)"],
    },
    "注意安全": {
        "category": "安全",
        "aliases": ["注意安全", "注意安全", "注意安全"],
        "patterns": [],
    },
    "登销记": {
        "category": "作业",
        "aliases": ["登销记", "登消记", "登小记"],
        "patterns": [r"登销记"],
    },
    "设备管理单位": {
        "category": "组织",
        "aliases": ["设备管理单位", "设别管理单位", "设备管理单为"],
        "patterns": [r"设备管理单位"],
    },
    "破封": {
        "category": "作业",
        "aliases": ["破封", "破风", "破峰"],
        "patterns": [r"破封开锁开箱"],
    },
    "开锁": {
        "category": "作业",
        "aliases": ["开锁", "开索", "开锁"],
        "patterns": [r"破封开锁开箱"],
    },
    "开箱": {
        "category": "作业",
        "aliases": ["开箱", "开乡", "开箱"],
        "patterns": [r"破封开锁开箱"],
    },
    "清点": {
        "category": "作业",
        "aliases": ["清点", "轻点", "清点"],
        "patterns": [r"清点数量核对编号"],
    },
    "核对编号": {
        "category": "作业",
        "aliases": ["核对编号", "合对编号", "核兑编号"],
        "patterns": [],
    },
    "共同": {
        "category": "用语",
        "aliases": ["共同", "共铜", "共动"],
        "patterns": [r"共同.*(清点|破封|试验)"],
    },
    "联合试验": {
        "category": "作业",
        "aliases": ["联合试验", "联和试验", "联合实验"],
        "patterns": [r"经.*联合试验良好"],
    },
    "恢复正常": {
        "category": "状态",
        "aliases": ["恢复正常", "恢复正常", "恢复正常"],
        "patterns": [r"恢复.*正常使用"],
    },
    "恢复使用": {
        "category": "状态",
        "aliases": ["恢复使用", "恢复使拥", "恢复使用"],
        "patterns": [],
    },
    "影响": {
        "category": "状态",
        "aliases": ["影响", "影向", "影饷"],
        "patterns": [r"影响.*(接发列车|调车作业)"],
    },
    "经由": {
        "category": "用语",
        "aliases": ["经由", "经油", "经尤"],
        "patterns": [r"经由该.*(道岔|区段)"],
    },
    "有关": {
        "category": "用语",
        "aliases": ["有关", "有光", "有关"],
        "patterns": [r"通知有关人员"],
    },
    "扳动": {
        "category": "操作",
        "aliases": ["扳动", "搬动", "板动"],
        "patterns": [r"来回扳动三次"],
    },
    "操纵": {
        "category": "操作",
        "aliases": ["操纵", "操众", "操从"],
        "patterns": [r"操纵.*道岔"],
    },
    "按压": {
        "category": "操作",
        "aliases": ["按压", "按亚", "安压"],
        "patterns": [r"按压.*按钮"],
    },
    "点击": {
        "category": "操作",
        "aliases": ["点击", "点机", "电击"],
        "patterns": [r"点击.*按钮"],
    },
    "排列": {
        "category": "操作",
        "aliases": ["正排", "整排", "真排"],
        "patterns": [r"正排.*信号"],
    },
    "开放": {
        "category": "操作",
        "aliases": ["开放", "开防", "开方"],
        "patterns": [r"开放信号", r"开放引导信号"],
    },
    "关闭": {
        "category": "操作",
        "aliases": ["关闭", "关必", "关毕"],
        "patterns": [r"关闭.*信号"],
    },
    "取消": {
        "category": "操作",
        "aliases": ["取消", "取销", "取肖"],
        "patterns": [r"取消.*进路"],
    },
    "变更": {
        "category": "操作",
        "aliases": ["变更", "变跟", "便更"],
        "patterns": [r"变更.*计划"],
    },
    "触发": {
        "category": "操作",
        "aliases": ["触发", "触法", "触乏"],
        "patterns": [r"触发方式改为(人工|自动)"],
    },
    "改方": {
        "category": "操作",
        "aliases": ["改方", "改芳", "改房"],
        "patterns": [r"办理.*改方"],
    },
    "点灯": {
        "category": "操作",
        "aliases": ["点灯", "点登", "点等"],
        "patterns": [r"(出站|进站)信号机.*点灯"],
    },
    "允许": {
        "category": "用语",
        "aliases": ["允许", "允需", "允序"],
        "patterns": [r"允许.*改方"],
    },
    "停电": {
        "category": "操作",
        "aliases": ["停电", "停点", "亭电"],
        "patterns": [r"接触网停电"],
    },
    "申请": {
        "category": "用语",
        "aliases": ["申请", "申情", "伸请"],
        "patterns": [r"申请.*(停电|开通|封锁)"],
    },
    "汇报": {
        "category": "用语",
        "aliases": ["汇报", "会报", "回抱"],
        "patterns": [r"接.*汇报", r"向.*汇报"],
    },
    "签认": {
        "category": "作业",
        "aliases": ["签认", "签任", "千认"],
        "patterns": [r"通知.*签认"],
    },
    "确认": {
        "category": "用语",
        "aliases": ["确认", "确人", "缺认"],
        "patterns": [r"确认.*(进路|信号|空闲)"],
    },
    "试验": {
        "category": "作业",
        "aliases": ["试验", "实验", "实雁"],
        "patterns": [r"联锁试验", r"联合试验"],
    },
    "良好": {
        "category": "状态",
        "aliases": ["良好", "粮好", "良号"],
        "patterns": [r"试验良好"],
    },
    "正常": {
        "category": "状态",
        "aliases": ["正常", "正长", "正尝"],
        "patterns": [r"设备正常", r"线路设备正常"],
    },
    "异常": {
        "category": "状态",
        "aliases": ["异常", "异长", "异尝"],
        "patterns": [r"线路无异常"],
    },
    "无异常": {
        "category": "状态",
        "aliases": ["无异常", "无异常", "无异常"],
        "patterns": [r"线路无异常", r"区段无异常"],
    },
    "完毕": {
        "category": "状态",
        "aliases": ["完毕", "完必", "完毕"],
        "patterns": [r"处理完毕", r"巡线完毕"],
    },
    "处理": {
        "category": "作业",
        "aliases": ["处理", "处里", "出理"],
        "patterns": [r"故障处理", r"设备处理"],
    },
    "修复": {
        "category": "状态",
        "aliases": ["修复", "修付", "休复"],
        "patterns": [r"故障已修复", r"恢复.*正常使用"],
    },
    "抢修": {
        "category": "作业",
        "aliases": ["抢修", "强修", "抢休"],
        "patterns": [r"立即抢修"],
    },
    "处理情况": {
        "category": "用语",
        "aliases": ["处理情况", "处里情况", "出理情况"],
        "patterns": [r"设备处理情况"],
    },
    "销记": {
        "category": "作业",
        "aliases": ["销记", "消记", "小记"],
        "patterns": [r"(工务|电务)销记"],
    },
    "登记": {
        "category": "作业",
        "aliases": ["登记", "登计", "登纪"],
        "patterns": [r"登记《.*》", r"电务登记"],
    },
    "填写": {
        "category": "作业",
        "aliases": ["填写", "添写", "天写"],
        "patterns": [r"填写《.*》", r"填写凭证"],
    },
    "抹消": {
        "category": "作业",
        "aliases": ["抹消", "末消", "抹销"],
        "patterns": [r"抹消占线簿"],
    },
    "撤回": {
        "category": "作业",
        "aliases": ["撤回", "撤会", "che回"],
        "patterns": [r"撤回加岗人员"],
    },
    "收回": {
        "category": "作业",
        "aliases": ["收回", "手回", "搜回"],
        "patterns": [r"收回.*(手摇把|钥匙)"],
    },
    "带至": {
        "category": "用语",
        "aliases": ["带至", "带至", "代至"],
        "patterns": [r"带至.*扳道房"],
    },
    "带到": {
        "category": "用语",
        "aliases": ["带到", "带道", "代到"],
        "patterns": [r"带到.*扳道房"],
    },
    "送至": {
        "category": "用语",
        "aliases": ["送至", "送致", "送之"],
        "patterns": [],
    },
    "到达": {
        "category": "状态",
        "aliases": ["到达", "倒达", "道达"],
        "patterns": [r"列车到达", r"到达立岗地点"],
    },
    "通过": {
        "category": "状态",
        "aliases": ["通过", "通国", "通个"],
        "patterns": [r"列车通过", r"通过信号"],
    },
    "出发": {
        "category": "状态",
        "aliases": ["出发", "初发", "出法"],
        "patterns": [r"待出发"],
    },
    "到达": {
        "category": "状态",
        "aliases": ["到达", "倒达", "道达"],
        "patterns": [r"到达.*站"],
    },
    "整列": {
        "category": "状态",
        "aliases": ["整列", "整烈", "正列"],
        "patterns": [r"整列到达"],
    },
    "尾部": {
        "category": "位置",
        "aliases": ["尾部", "尾不", "伟部"],
        "patterns": [r"列车尾部", r"尾部过标", r"尾部.*车"],
    },
    "头部": {
        "category": "位置",
        "aliases": ["头部", "头不", "投部"],
        "patterns": [r"列车头部"],
    },
    "前方": {
        "category": "位置",
        "aliases": ["前方", "钱方", "前芳"],
        "patterns": [r"前方区间"],
    },
    "后方": {
        "category": "位置",
        "aliases": ["后方", "后芳", "后防"],
        "patterns": [],
    },
    "内方": {
        "category": "位置",
        "aliases": ["内方", "内芳", "那方"],
        "patterns": [r"信号机内方"],
    },
    "外方": {
        "category": "位置",
        "aliases": ["外方", "外芳", "歪方"],
        "patterns": [r"信号机外方"],
    },
    "区间": {
        "category": "位置",
        "aliases": ["区间", "区见", "去间"],
        "patterns": [r"(上行|下行)区间", r"区间封锁", r"区间.*故障"],
    },
    "站界": {
        "category": "位置",
        "aliases": ["站界", "站接", "占界"],
        "patterns": [],
    },
    "警冲标": {
        "category": "设施",
        "aliases": ["警冲标", "警冲彪", "警冲表"],
        "patterns": [],
    },
    "绝缘节": {
        "category": "设施",
        "aliases": ["绝缘节", "绝缘结", "绝元节"],
        "patterns": [],
    },
    "应答器": {
        "category": "设备",
        "aliases": ["应答器", "应达器", "印答器"],
        "patterns": [],
    },
    "轨道电路": {
        "category": "设备",
        "aliases": ["轨道电路", "轨道电炉", "轨道店路"],
        "patterns": [r"轨道电路.*故障"],
    },
    "转辙机": {
        "category": "设备",
        "aliases": ["转辙机", "转撤机", "转折机"],
        "patterns": [r"转辙机钥匙"],
    },
    "信号楼": {
        "category": "设施",
        "aliases": ["信号楼", "新号楼", "信后楼"],
        "patterns": [r"信号楼.*值班员"],
    },
    "调度所": {
        "category": "设施",
        "aliases": ["调度所", "条度所", "调渡所"],
        "patterns": [r"调度所.*调度员"],
    },
    "指挥中心": {
        "category": "设施",
        "aliases": ["指挥中心", "指挥中辛", "只挥中心"],
        "patterns": [r"安全生产指挥中心", r"站段.*指挥中心"],
    },
    "机务段": {
        "category": "设施",
        "aliases": ["机务段", "机无段", "机物段"],
        "patterns": [],
    },
    "车辆段": {
        "category": "设施",
        "aliases": ["车辆段", "车量段", "车列段"],
        "patterns": [],
    },
    "工务段": {
        "category": "设施",
        "aliases": ["工务段", "公务段", "工无段"],
        "patterns": [],
    },
    "电务段": {
        "category": "设施",
        "aliases": ["电务段", "店务段", "电无段"],
        "patterns": [],
    },
    "供电段": {
        "category": "设施",
        "aliases": ["供电段", "供点段", "功电段"],
        "patterns": [],
    },
    "房建段": {
        "category": "设施",
        "aliases": ["房建段", "房见段", "防建段"],
        "patterns": [],
    },
    "通信段": {
        "category": "设施",
        "aliases": ["通信段", "通星段", "通信断"],
        "patterns": [],
    },
    "车务段": {
        "category": "设施",
        "aliases": ["车务段", "车无段", "彻务段"],
        "patterns": [],
    },
    "客运段": {
        "category": "设施",
        "aliases": ["客运段", "客晕段", "客运断"],
        "patterns": [],
    },
    "动车段": {
        "category": "设施",
        "aliases": ["动车段", "洞车段", "东车段"],
        "patterns": [],
    },
    "动车所": {
        "category": "设施",
        "aliases": ["动车所", "洞车所", "东车所"],
        "patterns": [r"热备动车组"],
    },
    "存车场": {
        "category": "设施",
        "aliases": ["存车场", "存车厂", "村车场"],
        "patterns": [],
    },
    "编组站": {
        "category": "设施",
        "aliases": ["编组站", "变组站", "扁组站"],
        "patterns": [],
    },
    "区段站": {
        "category": "设施",
        "aliases": ["区段站", "区断站", "去段站"],
        "patterns": [],
    },
    "会让站": {
        "category": "设施",
        "aliases": ["会让站", "会浪站", "汇让站"],
        "patterns": [],
    },
    "越行站": {
        "category": "设施",
        "aliases": ["越行站", "月行站", "粤行站"],
        "patterns": [],
    },
    "中间站": {
        "category": "设施",
        "aliases": ["中间站", "中见站", "中坚站"],
        "patterns": [],
    },
    "客运站": {
        "category": "设施",
        "aliases": ["客运站", "客晕站", "客运占"],
        "patterns": [],
    },
    "货运站": {
        "category": "设施",
        "aliases": ["货运站", "货晕站", "货运占"],
        "patterns": [],
    },
    "客货运站": {
        "category": "设施",
        "aliases": ["客货运站", "客货晕站", "客货运占"],
        "patterns": [],
    },
    "接轨站": {
        "category": "设施",
        "aliases": ["接轨站", "接鬼站", "接轨占"],
        "patterns": [],
    },
    "始发站": {
        "category": "设施",
        "aliases": ["始发站", "始法站", "始发占"],
        "patterns": [],
    },
    "终到站": {
        "category": "设施",
        "aliases": ["终到站", "中到站", "终道站"],
        "patterns": [],
    },
    "技术站": {
        "category": "设施",
        "aliases": ["技术站", "技术占", "技术站"],
        "patterns": [],
    },
    "营业站": {
        "category": "设施",
        "aliases": ["营业站", "营业占", "营叶站"],
        "patterns": [],
    },
    "非营业站": {
        "category": "设施",
        "aliases": ["非营业站", "非营业占", "非营叶站"],
        "patterns": [],
    },
    "乘降所": {
        "category": "设施",
        "aliases": ["乘降所", "成降所", "乘将所"],
        "patterns": [],
    },
    "线路所": {
        "category": "设施",
        "aliases": ["线路所", "线路索", "仙路所"],
        "patterns": [],
    },
    "辅助所": {
        "category": "设施",
        "aliases": ["辅助所", "辅助索", "付助所"],
        "patterns": [],
    },
    "安全线": {
        "category": "设施",
        "aliases": ["安全线", "安全现", "安泉线"],
        "patterns": [],
    },
    "避难线": {
        "category": "设施",
        "aliases": ["避难线", "避难现", "避难线"],
        "patterns": [],
    },
    "段管线": {
        "category": "设施",
        "aliases": ["段管线", "段管现", "断管线"],
        "patterns": [],
    },
    "岔线": {
        "category": "设施",
        "aliases": ["岔线", "岔现", "差线"],
        "patterns": [],
    },
    "特别用途线": {
        "category": "设施",
        "aliases": ["特别用途线", "特别用途现", "特别用图线"],
        "patterns": [],
    },
    "正线": {
        "category": "设施",
        "aliases": ["正线", "正现", "正线"],
        "patterns": [r"正线.*通过"],
    },
    "站线": {
        "category": "设施",
        "aliases": ["站线", "占线", "站现"],
        "patterns": [r"站线.*(到发线|调车线)"],
    },
    "到发线": {
        "category": "设施",
        "aliases": ["到发线", "到发现", "道法线"],
        "patterns": [],
    },
    "调车线": {
        "category": "设施",
        "aliases": ["调车线", "调车现", "条车线"],
        "patterns": [],
    },
    "牵出线": {
        "category": "设施",
        "aliases": ["牵出线", "牵出现", "千出线"],
        "patterns": [],
    },
    "货物线": {
        "category": "设施",
        "aliases": ["货物线", "货物现", "货雾线"],
        "patterns": [],
    },
    "站内正线": {
        "category": "设施",
        "aliases": ["站内正线", "站内正现", "占内正线"],
        "patterns": [],
    },
    "站内线": {
        "category": "设施",
        "aliases": ["站内线", "占内线", "站内现"],
        "patterns": [],
    },
    "区间正线": {
        "category": "设施",
        "aliases": ["区间正线", "区间正现", "去间正线"],
        "patterns": [],
    },
    "专用线": {
        "category": "设施",
        "aliases": ["专用线", "专用现", "专拥线"],
        "patterns": [],
    },
    "机车走行线": {
        "category": "设施",
        "aliases": ["机车走行线", "机车走行现", "机车走形线"],
        "patterns": [],
    },
    "机待线": {
        "category": "设施",
        "aliases": ["机待线", "机待现", "机代线"],
        "patterns": [],
    },
    "安全线": {
        "category": "设施",
        "aliases": ["安全线", "安全现", "安泉线"],
        "patterns": [],
    },
    "避难线": {
        "category": "设施",
        "aliases": ["避难线", "避难现", "避难线"],
        "patterns": [],
    },
    "段管线": {
        "category": "设施",
        "aliases": ["段管线", "段管现", "断管线"],
        "patterns": [],
    },
    "岔线": {
        "category": "设施",
        "aliases": ["岔线", "岔现", "差线"],
        "patterns": [],
    },
    "特别用途线": {
        "category": "设施",
        "aliases": ["特别用途线", "特别用途现", "特别用图线"],
        "patterns": [],
    },
}


def generate_phonetic_aliases(word: str) -> list[str]:
    """基于拼音混淆规则自动生成别名候选."""
    # 声母混淆矩阵
    initial_confusion = {
        'zh': ['z', 'zh'],
        'z': ['z', 'zh'],
        'sh': ['s', 'sh'],
        's': ['s', 'sh'],
        'ch': ['c', 'ch'],
        'c': ['c', 'ch'],
        'n': ['n', 'l'],
        'l': ['n', 'l'],
        'f': ['f', 'h'],
        'h': ['f', 'h'],
        'r': ['r', 'l'],
    }
    # 韵母混淆矩阵
    final_confusion = {
        'an': ['an', 'ang'],
        'ang': ['an', 'ang'],
        'en': ['en', 'eng'],
        'eng': ['en', 'eng'],
        'in': ['in', 'ing'],
        'ing': ['in', 'ing'],
        'ian': ['ian', 'iang'],
        'iang': ['ian', 'iang'],
        'uan': ['uan', 'uang'],
        'uang': ['uan', 'uang'],
    }
    
    # 获取拼音
    pys = pypinyin.lazy_pinyin(word, style=pypinyin.Style.INITIALS)
    fys = pypinyin.lazy_pinyin(word, style=pypinyin.Style.FINALS)
    
    aliases = set()
    # 对每个字，尝试替换声母和韵母
    for i in range(len(pys)):
        py = pys[i]
        fy = fys[i]
        
        # 声母替换
        if py in initial_confusion:
            for new_py in initial_confusion[py]:
                if new_py != py:
                    new_pinyin = pys.copy()
                    new_pinyin[i] = new_py
                    # 尝试转回汉字（这个需要拼音->汉字的映射，较复杂）
                    # 简化处理：仅记录可能的拼音组合
                    pass
        
        # 韵母替换
        if fy in final_confusion:
            for new_fy in final_confusion[fy]:
                if new_fy != fy:
                    pass
    
    return list(aliases)


def build_lexicon_from_corpus(corpus_path: Path, top_k: int = 200) -> dict:
    """从语料中提取高频词，补充到术语库."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # 加载自定义词典确保术语不被切分
    for term in CORE_TERMS:
        jieba.add_word(term, freq=10000)
    
    word_counter = Counter()
    for text in texts:
        words = jieba.lcut(text)
        words = [w for w in words if len(w) >= 2 and not w.isdigit()]
        word_counter.update(words)
    
    # 取高频词
    frequent_words = [word for word, _ in word_counter.most_common(top_k)]
    
    # 合并到CORE_TERMS
    lexicon = {"terms": []}
    existing = set(CORE_TERMS.keys())
    
    for term, info in CORE_TERMS.items():
        lexicon["terms"].append({
            "canonical": term,
            "category": info["category"],
            "aliases": info["aliases"],
            "patterns": info["patterns"],
        })
    
    # 将高频但不在CORE_TERMS中的词作为补充术语（无别名）
    for word in frequent_words:
        if word not in existing and len(word) >= 2:
            lexicon["terms"].append({
                "canonical": word,
                "category": "auto_extracted",
                "aliases": [],
                "patterns": [],
            })
    
    return lexicon


def build_alias_map(lexicon: dict) -> dict:
    """构建别名->标准词的映射表."""
    alias_map = {}
    for term_info in lexicon["terms"]:
        canonical = term_info["canonical"]
        for alias in term_info["aliases"]:
            alias_map[alias] = canonical
    return alias_map


def main():
    project_root = Path(__file__).parent.parent
    corpus_file = project_root / "data" / "corpus" / "railway_corpus.txt"
    lexicon_dir = project_root / "data" / "lexicon"
    lexicon_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果语料已存在，从中提取补充术语
    if corpus_file.exists():
        lexicon = build_lexicon_from_corpus(corpus_file)
    else:
        # 使用预定义核心术语
        lexicon = {"terms": []}
        for term, info in CORE_TERMS.items():
            lexicon["terms"].append({
                "canonical": term,
                "category": info["category"],
                "aliases": info["aliases"],
                "patterns": info["patterns"],
            })
    
    # 保存术语库
    terms_file = lexicon_dir / "railway_terms.json"
    with open(terms_file, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)
    print(f"[build_lexicon] 术语库已保存: {terms_file} ({len(lexicon['terms'])} 条)")
    
    # 构建并保存别名映射
    alias_map = build_alias_map(lexicon)
    alias_file = lexicon_dir / "aliases.json"
    with open(alias_file, "w", encoding="utf-8") as f:
        json.dump(alias_map, f, ensure_ascii=False, indent=2)
    print(f"[build_lexicon] 别名映射已保存: {alias_file} ({len(alias_map)} 条)")
    
    # 构建拼音混淆集
    confusion = {
        "initial": {
            "zh": ["z"], "z": ["zh"],
            "sh": ["s"], "s": ["sh"],
            "ch": ["c"], "c": ["ch"],
            "n": ["l"], "l": ["n"],
            "f": ["h"], "h": ["f"],
            "r": ["l"], "l": ["r"],
        },
        "final": {
            "an": ["ang"], "ang": ["an"],
            "en": ["eng"], "eng": ["en"],
            "in": ["ing"], "ing": ["in"],
            "ian": ["iang"], "iang": ["ian"],
            "uan": ["uang"], "uang": ["uan"],
        },
        "whole": {
            "si": ["shi", "shi"], "shi": ["si", "shi"],
            "zi": ["zhi", "zhi"], "zhi": ["zi", "zhi"],
            "ci": ["chi", "chi"], "chi": ["ci", "chi"],
        }
    }
    confusion_file = lexicon_dir / "phonetic_confusion.json"
    with open(confusion_file, "w", encoding="utf-8") as f:
        json.dump(confusion, f, ensure_ascii=False, indent=2)
    print(f"[build_lexicon] 拼音混淆集已保存: {confusion_file}")


if __name__ == "__main__":
    main()
