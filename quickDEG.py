#!/usr/bin/env python
###author:CY Gao (nongxinshengxin)

import pandas as pd
import numpy as np
import logging
import os
import yaml
import argparse
import sys
import io

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

import matplotlib.pyplot as plt
import seaborn as sns

class StreamToLogger(io.TextIOBase):
    """将标准流重定向到 logging 的自定义类"""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, message):
        if message.strip():  # 过滤空消息
            # 处理换行符分割的消息
            self.buffer += message
            if '\n' in self.buffer:
                lines = self.buffer.split('\n')
                for line in lines[:-1]:  # 最后一个是未完成行
                    if line.strip():
                        self.logger.log(self.level, line.strip())
                self.buffer = lines[-1]  # 保留未完成行

    def flush(self):
        pass  # logging 模块自带缓冲处理

class DEGflow():
    def __init__(self,config_file):
        """
        唯一属性，config文件
        """
        self.config=self._load_config(config_file)
        self.original_stdout = sys.stdout  # 保存原始流
        self.original_stderr = sys.stderr
        self.logger = self._setup_logger()  # 初始化日志
        self._redirect_std_streams()       # 重定向标准流

    def _load_config(self, config_file):
        """
        load yaml configuration file
        param config_file: str, the path of the configuration file
        """
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)  
        except FileNotFoundError:
            raise
        except yaml.YAMLError as e:
            raise

    def _setup_logger(self):
        """配置 logging 并返回日志记录器"""
        logger = logging.getLogger("DEGflow")
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        OUTPUT_PATH = self.config["output_path"]
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        file_handler = logging.FileHandler(f"{OUTPUT_PATH}/task.log")
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _redirect_std_streams(self):
        """重定向标准输出和错误到 logging"""
        sys.stdout = StreamToLogger(
            self.logger, 
            level=logging.INFO
        )
        sys.stderr = StreamToLogger(
            self.logger, 
            level=logging.ERROR
        )
        plt.set_loglevel('info')

    def __del__(self):
        """析构时恢复原始流"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


    def read_mat(self):
        valid_extensions = ['.csv', '.xls', '.xlsx', '.txt', '.tsv']
        matfile=self.config["matrix"]
        file_ext = os.path.splitext(matfile)[-1].lower()
    
        if file_ext not in valid_extensions:
            self.logger.error(f"meta文件不支持的文件类型: {file_ext}，支持的类型: {valid_extensions}")
            raise ValueError
        
        read_params = {
            '.csv': {'method': pd.read_csv, 'kwargs': {'index_col': 0}},
            '.xls': {'method': pd.read_excel, 'kwargs': {'index_col': 0}},
            '.xlsx': {'method': pd.read_excel, 'kwargs': {'index_col': 0}},
            '.txt': {'method': pd.read_csv, 'kwargs': {'sep': '\t','index_col': 0}},
            '.tsv': {'method': pd.read_csv, 'kwargs': {'sep': '\t','index_col': 0}}
            }

        try:
            df = read_params[file_ext]['method'](
                matfile, 
                **read_params[file_ext]['kwargs']
            )
        except Exception as e:
            self.logger.error(f"matrix文件读取失败: {str(e)}")
            raise RuntimeError
        else:
            self.logger.info("matrix文件读取成功")


        # 检查是否存在缺失值
        if df.isnull().values.any():
            self.logger.error("matrix数据包含缺失值，不是标准矩阵")
            raise ValueError

        # 检查是否为二维结构
        if len(df.shape) != 2:
            self.logger.error("matrix数据不是二维结构")
            raise ValueError
        
        numeric_dtypes = {'int16', 'int32', 'int64'} ##创建一个集合set
    
        # 检查所有列是否为数值类型
        for dtype in df.dtypes:
            if str(dtype) not in numeric_dtypes:
                self.logger.error("matrix数据内存在非数值型元素，或数据内存在非整数，请检查文件")
                raise ValueError

        if (df.select_dtypes(include='number') < 0).any().any():
            self.logger.error("matrix数据内存在负值，请检查文件")
            raise ValueError
        
        return df
    

    def read_metafile(self):
        valid_extensions = ['.csv', '.xls', '.xlsx', '.txt', '.tsv']
        metafile=self.config["meta"]
        file_ext = os.path.splitext(metafile)[-1].lower()
    
        if file_ext not in valid_extensions:
            self.logger.error(f"meta文件不支持的文件类型: {file_ext}，支持的类型: {valid_extensions}")
            raise ValueError
        
        with open(metafile) as f:
            first_line = f.readline().strip()
        if "condition" in first_line:
            read_params = {
                '.csv': {'method': pd.read_csv, 'kwargs': {'index_col': 0}},
                '.xls': {'method': pd.read_excel, 'kwargs': {'index_col': 0}},
                '.xlsx': {'method': pd.read_excel, 'kwargs': {'index_col': 0}},
                '.txt': {'method': pd.read_csv, 'kwargs': {'sep': '\t','index_col': 0}},
                '.tsv': {'method': pd.read_csv, 'kwargs': {'sep': '\t','index_col': 0}}
                }
        else:
            read_params = {
                '.csv': {'method': pd.read_csv, 'kwargs': {'index_col': 0, 'names':['condition']}},
                '.xls': {'method': pd.read_excel, 'kwargs': {'index_col': 0, 'names':['condition']}},
                '.xlsx': {'method': pd.read_excel, 'kwargs': {'index_col': 0, 'names':['condition']}},
                '.txt': {'method': pd.read_csv, 'kwargs': {'sep': '\t','index_col': 0, 'names':['condition']}},
                '.tsv': {'method': pd.read_csv, 'kwargs': {'sep': '\t','index_col': 0, 'names':['condition']}}
                }
            self.logger.info("meta文件缺少表头condition，已为您自动添加")

        try:
            meta = read_params[file_ext]['method'](
                metafile, 
                **read_params[file_ext]['kwargs']
            )
        except Exception as e:
            self.logger.error(f"meta文件读取失败: {str(e)}")
            raise RuntimeError
        else:
            self.logger.info("meta文件读取成功")

        return meta
    
    def deg_analysis(self):
        df=self.read_mat()
        df=df.T
        meta=self.read_metafile()
        if not all(sample in df.index for sample in meta.index):
            self.logger.error("meta文件中sample名与matrix文件中sample名不一致")
            raise RuntimeError

        df=df.loc[meta.index]


        inference = DefaultInference(n_cpus=8)
        dds = DeseqDataSet(
            counts=df,
            metadata=meta,
            design="~condition",
            refit_cooks=True,
            inference=inference,
        )

        dds.deseq2()

        control=self.config["control"]
        treatment=self.config["treatment"]
        self.logger.info(f"注意：对照组为{control}，实验组为{treatment}")





        ds = DeseqStats(dds, contrast=["condition", treatment, control], inference=inference)
        #print(print(dir(ds)))
        #print(ds.summary().type)
        ds.summary()
        resultdf=ds.results_df
        return resultdf
    
    def deg_result_output_plot(self):
        resultdf=self.deg_analysis()
        OUTPUT_PATH=self.config["output_path"]

        resultdf['change'] = 'stable'
        resultdf.loc[(resultdf['log2FoldChange'] >= 1) & (resultdf['padj'] < 0.05), 'change'] = 'up'

        resultdf.loc[(resultdf['log2FoldChange'] <= -1) & (resultdf['padj'] < 0.05), 'change'] = 'down'
        resultdf.to_csv(os.path.join(OUTPUT_PATH, "all_results.csv"))

        resultdf=resultdf.filter(items=['log2FoldChange','pvalue','padj','change'])
        up_df=resultdf.loc[(resultdf['change']=='up')]
        down_df=resultdf.loc[(resultdf['change']=='down')]

        self.logger.info('上调表达基因与下调表达基因分别储存为up_results.csv和down_results.csv')
        up_df.to_csv(os.path.join(OUTPUT_PATH, "up_results.csv"))
        down_df.to_csv(os.path.join(OUTPUT_PATH, "down_results.csv"))
        


        resultdf.dropna(how='all',inplace=True)
        y_max = max(-np.log10(resultdf['padj'] + 1e-10))
        #resultdf['neg_log_pvalue'] = -np.log10(resultdf['pvalue'] + 1e-10)

        ####plot
        sns.set_style("whitegrid")
        #plt.rcParams['font.family'] = 'Arial'  

        self.logger.info('开始绘制火山图......')
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        sns.scatterplot(
            data=resultdf,
            x='log2FoldChange',
            y=-np.log10(resultdf['padj'] + 1e-10),
            hue='change',
            palette={'up': '#E64B35', 'down': '#3182BD', 'stable': 'grey'}, 
            alpha=0.7,  
            s=40,       
            edgecolor='none',
            ax=ax
            )

        ax.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=1, alpha=0.8)  
        ax.axvline(-1, color='red', linestyle='--', linewidth=1, alpha=0.8)
        ax.axvline(1, color='red', linestyle='--', linewidth=1, alpha=0.8)

        plt.title("Volcano Plot (Differential Expression)", fontsize=14, pad=20)
        plt.xlabel("log2(Fold Change)", fontsize=12)
        plt.ylabel("-log10(p-adjusted)", fontsize=12)
        plt.legend(title='Change', loc='upper right', frameon=True)

        plt.xlim(resultdf['log2FoldChange'].min()*1.1, resultdf['log2FoldChange'].max()*1.1)
        plt.ylim(0, y_max)


        plt.savefig(f'{OUTPUT_PATH}/volcano_plot.pdf', dpi=300, bbox_inches='tight')
        self.logger.info('火山图绘制完毕，保存为pdf文件')
        #plt.show()




 

def main():
    parser = argparse.ArgumentParser(description='Count the density of features in bins')
    parser.add_argument('config', help='Input file location')
    args = parser.parse_args()

    try:
        deg = DEGflow(args.config)
        deg.deg_result_output_plot()
    except Exception as e:
        logging.getLogger("DEGflow").critical(f"流程异常终止: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()