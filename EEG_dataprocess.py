import mne
import matplotlib.pyplot as plt
import numpy as np
import os
import pyedflib


"""
    初步了解edf文件,了解脑电波数据
"""



# 加载edf文件
def load_edf_file(filepath):
    """
    加载EDF文件并返回原始数据对象
    :return:raw_data
    """
    try:
        # 读取EDF文件
        raw = mne.io.read_raw_edf(filepath, preload=True)
        return raw
    except Exception as e:
        print(f"加载EDF文件时出错: {e}")
        return None

# 滤波,1-45hz的滤波
def edf_file_filter(raw):
    """
    对EDF文件原始数据进行1-45hz的fir滤波处理
    参数:
        raw: mne.io.Raw对象，EDF文件的原始数据
    返回:
        filtered_raw: 滤波后的Raw对象
    """
    if raw is None:
        print("警告: 输入的raw数据为空!")
        return None
    try:
        filtered_raw = raw.copy()
        filtered_raw.filter(l_freq=1.0, h_freq=45.0, method = 'fir', fir_window='hamming')
        return filtered_raw
    except Exception as e:
        print(f"滤波处理EDF文件时出错: {e}")
        return None

# 减少通道数,从64减到16
def decrease_channels(raw):
    """
    对EDF文件原始数据进行选取16个通道处理。

    参数:
        raw: mne.io.Raw对象, EDF文件的原始数据。

    返回:
        selected_data: 选择特定通道后的Raw对象。
    """
    if raw is None:
        print("警告: 输入的raw数据为空!")
        return None
    try:
        # 目标通道列表
        target_channels = ['Fc1.', 'C4..', 'T10.', 'Tp8.', 'Po8.',
                           'Af7.', 'Af3.', 'Af4.', 'Af8.', 'F1..', 'F6..', 'Ft8.', 'Tp7.', 'P1..', 'O2..',
                           'T7..']

        # 提取目标通道的数据
        selected_data = raw.copy().pick(target_channels)
        return selected_data
    except Exception as e:
        print(f"滤波处理EDF文件时出错: {e}")
        return None

# 保存函数
def save_processed_data(raw, output_path):
    """
    保存处理后的数据
    """
    if raw is None:
        print("警告: 输入的raw数据为空!")
        return

    try:
        raw.export(output_path, fmt='edf')
        print(f"成功保存到: {output_path}")
        return

    except Exception as e:
        raise RuntimeError(f"保存失败: {str(e)}")

# 绘制脑电波数据图
def edf_file_plot(raw):
    """
    绘制 EDF 文件原始数据
    :return: plot图像
    """
    if raw is None:
        print("警告: 输入的raw数据为空!")
        return
    try:
        # 打印基本信息
        # print(raw.info)  # 显示采样率、通道数等信息
        # 绘制原始数据（交互式窗口）
        raw.plot(block=True)

    except Exception as e:
        print(f"绘制数据plot图像出错: {e}")

# 绘制功率谱密度图
def edf_file_plot_psd(raw):
    """
    绘制 EDF文件的功率谱密度(PSD)图像
    :return: none
    """
    if raw is None:
        return
    try:
        # 绘制原始数据（交互式窗口）
        raw.plot_psd(area_mode='range', average=True, dB=True)
    except Exception as e:
        print(f"绘制数据psd_plot图像出错: {e}")

# 将数据分为总时长60s的分段
def segment_edf_one_eeg(edf_path, segment_length=0.5, sample_rate=160):
    """
    提取0-60秒内的数据并按固定时长分段

    参数:
        edf_path: 输入EDF文件路径
        segment_length: 分段时长(秒)，默认0.5秒
        output_path: 如需保存分段数据为NPZ文件则指定路径

    返回:
        (segments, channel_names) 或 (None, None) 如果失败
        segments: 形状为(段数, 通道数, 每段点数)的numpy数组
        channel_names: 所有通道名称列表
    """
    try:
        with pyedflib.EdfReader(edf_path) as f:
            # 获取所有通道信息
            channel_names = f.getSignalLabels()
            n_channels = f.signals_in_file
            total_samples = f.getNSamples()[0] # 值为9760

            # 计算0-60秒对应的采样点范围
            target_duration = 60  # 秒
            max_samples = int(target_duration * sample_rate) # 值为9600

            # 读取0-60秒数据
            eeg_data = np.zeros((n_channels, max_samples))
            for i in range(n_channels):
                eeg_data[i] = f.readSignal(i, 0, max_samples)


            # 计算分段参数
            points_per_segment = int(segment_length * sample_rate)
            if points_per_segment <= 0:
                raise ValueError("分段点数必须为正整数")

            n_segments = max_samples // points_per_segment
            if n_segments == 0:
                raise ValueError("数据长度不足以分割，请减小segment_length")

            # 执行分割
            segments = np.zeros((n_segments, n_channels, points_per_segment))
            for seg_idx in range(n_segments):
                start = seg_idx * points_per_segment
                end = start + points_per_segment
                segments[seg_idx] = eeg_data[:, start:end]

            return segments, channel_names

    except Exception as e:
        print(f"分割EDF文件出错: {str(e)}")
        return None, None


# 将数据分为总时长120s的分段
def segment_edf_two_eeg(edf_path, segment_length=0.5, sample_rate=160):
    """
    提取0-120秒内的数据并按固定时长分段

    参数:
        edf_path: 输入EDF文件路径
        segment_length: 分段时长(秒)，默认0.5秒
        output_path: 如需保存分段数据为NPZ文件则指定路径

    返回:
        (segments, channel_names) 或 (None, None) 如果失败
        segments: 形状为(段数, 通道数, 每段点数)的numpy数组
        channel_names: 所有通道名称列表
    """
    try:
        with pyedflib.EdfReader(edf_path) as f:
            # 获取所有通道信息
            channel_names = f.getSignalLabels()
            n_channels = f.signals_in_file
            total_samples = f.getNSamples()[0]

            # 计算0-120秒对应的采样点范围
            target_duration = 120  # 120秒
            max_samples = int(target_duration * sample_rate) # 值为9600*2

            # 读取0-120秒数据
            eeg_data = np.zeros((n_channels, max_samples))
            for i in range(n_channels):
                eeg_data[i] = f.readSignal(i, 0, max_samples)


            # 计算分段参数
            points_per_segment = int(segment_length * sample_rate)
            if points_per_segment <= 0:
                raise ValueError("分段点数必须为正整数")

            n_segments = max_samples // points_per_segment
            if n_segments == 0:
                raise ValueError("数据长度不足以分割，请减小segment_length")

            # 执行分割
            segments = np.zeros((n_segments, n_channels, points_per_segment))
            for seg_idx in range(n_segments):
                start = seg_idx * points_per_segment
                end = start + points_per_segment
                segments[seg_idx] = eeg_data[:, start:end]

            return segments, channel_names

    except Exception as e:
        print(f"分割EDF文件出错: {str(e)}")
        return None, None



# 将数据分为120s的分段,采样率自动检测文件中所给的采样率,而不是取一个恒定值
def segment_edf_eeg(edf_path, segment_length=0.5):
    """
    提取0-120秒内的数据并按固定时长分段

    参数:
        edf_path: 输入EDF文件路径
        segment_length: 分段时长(秒)，默认0.5秒
        output_path: 如需保存分段数据为NPZ文件则指定路径

    返回:
        (segments, channel_names) 或 (None, None) 如果失败
        segments: 形状为(段数, 通道数, 每段点数)的numpy数组
        channel_names: 所有通道名称列表
    """
    try:
        with pyedflib.EdfReader(edf_path) as f:
            # 获取所有通道信息

            channel_names = f.getSignalLabels()
            n_channels = f.signals_in_file
            total_samples = f.getNSamples()[0]
            sample_rate = f.getSampleFrequency(0)
            # 计算0-120秒对应的采样点范围
            target_duration = 120  # 120秒
            max_samples = int(target_duration * sample_rate) # 值为9600*2

            # 读取0-120秒数据
            eeg_data = np.zeros((n_channels, max_samples))
            for i in range(n_channels):
                eeg_data[i] = f.readSignal(i, 0, max_samples)


            # 计算分段参数
            points_per_segment = int(segment_length * sample_rate)
            if points_per_segment <= 0:
                raise ValueError("分段点数必须为正整数")

            n_segments = max_samples // points_per_segment
            if n_segments == 0:
                raise ValueError("数据长度不足以分割，请减小segment_length")

            # 执行分割
            segments = np.zeros((n_segments, n_channels, points_per_segment))
            for seg_idx in range(n_segments):
                start = seg_idx * points_per_segment
                end = start + points_per_segment
                segments[seg_idx] = eeg_data[:, start:end]

            return segments, channel_names

    except Exception as e:
        print(f"分割EDF文件出错: {str(e)}")
        return None, None



# 按照文件本身时长进行0.5s的分割
def segment_edf_all_eeg(edf_path, segment_length=0.5, sample_rate=160):
    """
    这个函数只给S106R05.edf和S104R08.edf使用
    """
    try:
        with pyedflib.EdfReader(edf_path) as f:
            # 获取所有通道信息
            channel_names = f.getSignalLabels()
            n_channels = f.signals_in_file
            total_samples = f.getNSamples()[0]
            target_duration = f.getFileDuration()
            # 计算0-60秒对应的采样点范围
            max_samples = int(target_duration * sample_rate)

            # 读取0-60秒数据
            eeg_data = np.zeros((n_channels, max_samples))
            for i in range(n_channels):
                eeg_data[i] = f.readSignal(i, 0, max_samples)


            # 计算分段参数
            points_per_segment = int(segment_length * sample_rate)
            if points_per_segment <= 0:
                raise ValueError("分段点数必须为正整数")

            n_segments = max_samples // points_per_segment
            if n_segments == 0:
                raise ValueError("数据长度不足以分割，请减小segment_length")

            # 执行分割
            segments = np.zeros((n_segments, n_channels, points_per_segment))
            for seg_idx in range(n_segments):
                start = seg_idx * points_per_segment
                end = start + points_per_segment
                segments[seg_idx] = eeg_data[:, start:end]

            return segments, channel_names

    except Exception as e:
        print(f"分割EDF文件出错: {str(e)}")
        return None, None



# 主程序
if __name__ == "__main__":

    ''' 
    1.初步处理EDF文件,初步了解EDF数据的格式
    import mne
    import matplotlib.pyplot as plt
    '''
    # # 输入EDF文件路径
    # edf_file = "./files/S106/S106R05.edf"
    # output_file = "filter"
    #
    # print(f"正在加载EDF文件: {edf_file}")
    # raw_data = load_edf_file(edf_file)
    # import pyedflib
    #
    #
    #
    # try:
    #     # 打开EDF文件
    #     with pyedflib.EdfReader(edf_file) as f:
    #         # 获取基本信息
    #         n_channels = f.signals_in_file  # 通道数量
    #         signal_labels = f.getSignalLabels()  # 信号标签
    #         sample_frequencies = f.getSampleFrequency(0) if n_channels > 0 else None  # 第一个信号的采样频率，可以遍历所有信号
    #         # 更准确的方法是获取每个信号的采样频率
    #         sample_frequencies = [f.getSampleFrequency(i) for i in range(n_channels)]
    #         durations = [f.getFileDuration()]  # 文件持续时间（秒）
    #         startdate = f.getStartdatetime()  # 开始日期和时间
    #
    #
    #         print(f"文件路径: {edf_file}")
    #         print(f"通道数量: {n_channels}")
    #         print(f"信号标签: {signal_labels}")
    #         print(f"采样频率 (Hz): {sample_frequencies}")
    #         print(f"文件持续时间 (秒): {durations[0]}")
    #         print(f"开始日期和时间: {startdate}")
    #
    #
    # except Exception as e:
    #     print(f"无法加载EDF文件: {e}")
    #
    # # 1hz-45hz的滤波处理
    # processed_data= edf_file_filter(raw_data)
    # print(processed_data.info)

    # # 保存第一个滤波处理的文件
    # save_processed_data(processed_data, output_file)
    #
    # # 绘制 PSD（限制频率范围，确保对数坐标）
    # edf_file_plot_psd(raw_data)
    # plt.show()

    """
    2.for循环遍历滤波处理前五个实验者的14种脑电数据
    import os
    import mne
    """
    # # 定义输入和输出目录路径
    # folder_root = 'files\S007'
    # output_root = "filter\S007"
    #
    # # 确保输出目录存在（如果不存在则自动创建）
    # os.makedirs(output_root, exist_ok=True)  # 关键添加：自动创建目录
    #
    # # 遍历指定目录下的所有EDF文件
    # for file_name in os.listdir(folder_root):
    #     if file_name.endswith('.edf'):
    #         # 构建当前EDF文件的完整路径
    #         edf_file_path = os.path.join(folder_root, file_name)
    #
    #         # 构建处理后数据的保存路径
    #         output_file = os.path.join(output_root, file_name)
    #
    #         # 读取EDF文件中的原始数据
    #         raw_data = load_edf_file(edf_file_path)
    #
    #         # 对原始数据进行滤波处理
    #         processed_data = edf_file_filter(raw_data)
    #
    #         # 保存
    #         save_processed_data(processed_data, output_file)

    """
    3.对滤波处理后的EDF数据进行通道筛选,将原来的64通道减少到必要的16通道,后续再考虑64通道
    """
    # # 定义输入和输出目录路径
    # file_root = 'filter\S006'
    # output_root = "16_channels\S006"
    # # 遍历指定目录下的所有EDF文件
    # for file_name in os.listdir(file_root):
    #     if file_name.endswith('.edf'):
    #         # 构建当前EDF文件的完整路径
    #         edf_file_path = os.path.join(file_root, file_name)
    #
    #         # 构建处理后数据的保存路径
    #         output_file = os.path.join(output_root, file_name)
    #
    #         # 读取EDF文件中的原始数据
    #         raw_data = load_edf_file(edf_file_path)
    #
    #         # 对原始数据进行16通道筛选处理
    #         processed_data = decrease_channels(raw_data)
    #
    #         # 保存
    #         save_processed_data(processed_data, output_file)




    """
      4.将每个通道的数据进行数据分割,按0.5s的指定时长进行切割
      序号88,92,100的EEG数据出现问题,无法分割3-14的EDF文件(采样率的问题,这三者的采样率不为恒定的160hz,改用检测采样率后问题解决)
      S104R08.edf(文件只有106秒)与S106R05.edf(文件只有37秒)出错
    """
    # # 定义输入和输出目录路径
    # edf_file_path_root = 'files'
    # output_file_root = "64_channels_seg"
    # subjects = [f"S{i:03d}" for i in range(7, 110)]
    # print(subjects)
    # subjects = ["S092"]
    # subjects = ["S001", "S002", "S003", "S004", "S005"]





    # 处理EDF文件并保存分段数据
    # try:
    #     data_seg, channels = segment_edf_two_eeg(edf_path)
    #     np.savez(os.path.join(output_dir, f"{subject}R{run:02d}.npz"), *data_seg)
    #     print(f"处理成功: {edf_path} -> {output_dir}/{subject}R{run:02d}.npz")
    # except Exception as e:
    #     print(f"处理失败 {edf_path}: {str(e)}")
    #
    # subjects = [f"S{i:03d}" for i in range(1, 110)]
    # runs = range(3, 15) # 在这里选择R01-R02还是R03-R14,注意seg分割函数也要改
    # for subject in subjects:
    #     for run in runs:
    #         # 生成EDF文件名
    #         edf_name = f"{subject}R{run:02d}.edf"  # 格式化为R01, R02,...R14
    #         edf_path = os.path.join(edf_file_path_root, subject, edf_name)
    #
    #         # 跳过不存在的文件
    #         if not os.path.exists(edf_path):
    #             print(f"文件不存在，已跳过: {edf_path}")
    #             continue
    #
    #         # 创建输出目录
    #         output_dir = output_file_root
    #
    #
    #         # 处理EDF文件并保存分段数据
    #         try:
    #             data_seg, channels = segment_edf_two_eeg(edf_path)
    #             np.savez(os.path.join(output_dir, f"{subject}R{run:02d}.npz"), *data_seg)
    #             print(f"处理成功: {edf_path} -> {output_dir}/{subject}R{run:02d}.npz")
    #         except Exception as e:
    #             print(f"处理失败 {edf_path}: {str(e)}")


    # 验证分割结果是否正确
    # 注意:在一个实验者中,R01和R02时长为1min,其余均为2min,故函数切割的调用需注意
    edf_file = "./16_channels/S001/S001R01.edf"
    data_seg, channels = segment_edf_one_eeg(edf_file)

    npz_file= "./16_channels_seg/S001R01.npz"
    data = np.load(npz_file)
    features = []
    a = 0
    for key in data.files:
        eeg = data[key]  # shape=(16,80)
        print(eeg)
        print(eeg.shape)

        a += 1
        if a == 4:
            break


    # a = 0
    # npz_file= "./16_channels_seg/S001R01.npz"
    # data = np.load(npz_file)
    # print("="*25)
    # print(data.files)  # 打印文件中的键
    # for i in data.files:
    #     print(i)
    #     print(data[i])  # 打印分割后的数据
    #
    #     a += 1
    #     if a == 4:
    #         break


    # # 保存为npz文件示例
    # save_dir = r"seg_Data/example"
    #
    # for i, segment in enumerate(data_seg):
    #     np.savez(os.path.join(save_dir, "S001R01.npz"), *data_seg)





    """
    专门处理两个无法分割的文件S106R05.edf和S104R08.edf
    """
    # subject="S104"
    # edf_name = "S104R08.edf"  # 格式化为R01, R02,...R14
    # edf_path = "files/S104/S104R08.edf"
    #
    # # 创建输出目录
    # output_dir = output_file_root
    # # 处理EDF文件并保存分段数据
    #
    # try:
    #     data_seg, channels = segment_edf_all_eeg(edf_path)
    #     np.savez(os.path.join(output_dir, "{S104R08.npz"), *data_seg)
    #     print(f"处理成功: {edf_path} -> {output_dir}/S104R08.npz")
    # except Exception as e:
    #     print(f"处理失败 {edf_path}: {str(e)}")


    """
        5.查看npz文件数据
    """



    data_dir = "16_channels_seg"
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]


    num = 0
    # 加载所有NPZ文件数据
    X, y = [], []
    for file in npz_files:
        data = np.load(os.path.join(data_dir, file))
        if num == 5:
            break
        for key in data.files:
            X.append(data[key].shape)# 添加通道维度
            y.append(file.split('S')[1].split('R')[0])  # 从文件名提取标签
            num = num+1
            if num ==5:
                break

        data.close()


    print(X[1])
    print(y)