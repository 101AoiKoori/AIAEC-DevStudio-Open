# 八项视觉识别任务文献

条目按本项目用来解释的设计问题归类。外部研究提供方法依据，项目数据与结果见[各任务设计](../vision-tasks/README.md)。

## 知觉分组与捷径学习

- Wagemans 等，2012，*A Century of Gestalt Psychology in Visual Perception I: Perceptual Grouping and Figure–Ground Organization*。用图形与背景、邻近和轮廓连续性分析错误分组和实例归属。该文为视觉科学综述，联系到本项目的样例属于设计解释。[Psychological Bulletin 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3482144/)
- Geirhos 等，2020，*Shortcut Learning in Deep Neural Networks*。解释训练分布内有效、条件变化后失效的预测线索。项目用背景交换、桶色与状态交叉、空场反例等设计检查这种依赖。[Nature Machine Intelligence](https://doi.org/10.1038/s42256-020-00257-z)

## 渲染、合成与成像

- Debevec，1998，*Rendering Synthetic Objects into Real Scenes*。用环境光照、场景代理和合成建立前景与实景间的光照联系，支撑项目对影子、接触面与曝光一致性的关注。[SIGGRAPH](https://doi.org/10.1145/280814.280864)
- Dwibedi、Misra、Hebert，2017，*Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection*。实例裁贴可扩展背景与位置，适用于立牌和贴纸的低成本候选生成。纸面材质与遮挡的适用范围仍由项目对照检查。[ICCV](https://openaccess.thecvf.com/content_iccv_2017/html/Dwibedi_Cut_Paste_and_ICCV_2017_paper.html)
- Tobin 等，2017，*Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*。通过变化渲染条件研究模拟到真实的迁移，作为光照、视角与外观组合设计的参考。[作者论文](https://arxiv.org/abs/1703.06907)
- Brooks 等，2019，*Unprocessing Images for Learned Raw Denoising*。研究相机处理链和传感器噪声在合成训练输入中的影响，说明同一种亮度调整难以代表真实相机成像。项目中的色调匹配是针对场景的近似。[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.html)
- Ghiasi 等，2021，*Simple Copy-Paste Is a Strong Data Augmentation Method for Instance Segmentation*。实例掩膜可随裁贴更新。用于理解人物、贴纸与标志的实例标签维护和遮挡对照。[CVPR 作者论文](https://arxiv.org/abs/2012.07177)
- Tremblay 等，2018，*Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization*。随机变化照明、姿态与纹理，提供面向检测任务的合成域随机化对照。[CVPR Workshops 作者论文](https://arxiv.org/abs/1804.06516)

## 纸面变形与文字识别

- Ma 等，2018，*DocUNet: Document Image Unwarping via a Stacked U-Net*。把折弯纸面的恢复建模为稠密变形问题，可用于解释楼名纸条上单一平面透视校正的局限。[CVPR](https://openaccess.thecvf.com/content_cvpr_2018/html/Ma_DocUNet_Document_Image_CVPR_2018_paper.html)
- Jaderberg 等，2014，*Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition*。用合成训练数据研究场景文字转录，为字面、背景和成像扰动的生成设计提供参考。楼名定位和 OCR 转录应分别验证。[作者论文](https://arxiv.org/abs/1406.2227)

## 实例计数与重叠抑制

- Bodla 等，2017，*Soft-NMS: Improving Object Detection With One Line of Code*。以重叠框分数衰减替代直接抑制，作为电动车密集遮挡失效的对照方法。[ICCV 作者论文](https://arxiv.org/abs/1704.04503)
- Chu 等，2020，*Detection in Crowded Scenes: One Proposal, Multiple Predictions*。研究重叠实例的多预测，为区分远端合框与后处理中误删单实例提供思路。[CVPR 作者论文](https://arxiv.org/abs/2003.09163)
- He 等，2017，*Mask R-CNN*。其空间对齐设计说明微小特征受裁块采样影响。桶开闭篇中的原图高分辨率裁块是本项目独立实现。[ICCV 作者论文](https://arxiv.org/abs/1703.06870)

## 投影几何与指针读数

- Zhang，2000，*A Flexible New Technique for Camera Calibration*。通过多姿态平面求取相机参数并建模镜头畸变，为墙面、道路和表盘的几何标定提供依据。[IEEE TPAMI 作者页面](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/)
- Milana 等，2022，*Autonomous Reading of Gauges in Unstructured Environments*。把表盘定位、视角校正和指针读数拆成独立环节，可对照仪表篇的几何读数链与误差来源。[Sensors](https://www.mdpi.com/1424-8220/22/17/6681)
