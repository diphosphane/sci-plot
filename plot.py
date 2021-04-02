#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt
plt.switch_backend('agg')
import numpy as np
import yaml
import os
import sys
from typing import Callable, Dict, Iterable, Iterator, Optional, Pattern, Tuple, List, TypeVar, Union, Any, Sequence
from functools import reduce
from dataclasses import dataclass

# todo: asymptotic line
# todo: axies text line adjust
# todo: adjust template style
# todo: support rcParam
# todo: support image cut
# todo: support CLI
# todo: support CLI to yaml convert
# todo: template output
# todo: switch backend: agg ps pdf::::  no x11 -> agg + imgcat   x11  -> plt.show + imgcat 

_num = Union[int, float]
_str_num = Union[_num, str]

class StringConst():
    ### fig string ###
    fig = 'fig'
    title = 'title'
    x_label = 'x_label'
    y_label = 'y_label'
    x_limit = 'x_limit'
    y_limit = 'y_limit'
    size = 'size'
    out_file = 'out_file'
    # todo: legend_loc
    # todo: scale for x,y axies
    ### plot string ###
    file = 'file'
    file_name = 'file_name'
    plot = 'plot'
    type = 'type'
    scatter  = 'scatter'
    xy_idx = 'xy_idx'
    label = 'label'
    style = 'style'
    color = 'color'
    line  = 'line'
    marker = 'marker'
    text = 'text'
    alpha = 'alpha'
    # scatter string
    facecolor = 'facecolor'
    linewidth = 'linewidth'
    # variable
    x2y = 'x2y'
    bias = 'bias'
    
    
class PlotConfig():
    s = StringConst
    def __init__(self, yaml_obj) -> None:
        s = self.s
        # global
        self.fig: dict = yaml_obj[s.fig]
        self.file: dict = yaml_obj[s.file]
        self.fig_config_read()
    
    def fig_config_read(self):
        s = self.s
        self.title: str = self.fig[s.title]
        self.x_label = self.fig.get(s.x_label)
        self.y_label = self.fig.get(s.y_label)
        self.x_limit = self.fig.get(s.x_limit)
        self.y_limit = self.fig.get(s.y_limit)
        self.size = self.fig.get(s.size)
        self.out_file = self.fig.get(s.out_file)
        
    def get(self, file_idx: int, prop: str):
        file = self.file[file_idx]
        return file.get(prop)


class DataFrame():
    def __init__(self, data: Sequence[float]) -> None:
        self.data = np.array(data)

    def __add__(self, bias: Union[float, Sequence[float]]) -> 'DataFrame':
        if bias:
            return self.__class__(self.data + bias)
        else:
            return self

    @staticmethod
    def MAE(one: 'DataFrame', other: 'DataFrame', string: bool=False) -> Union[str, float]:
        err = abs(one.data - other.data)
        mae = reduce(lambda x,y: x+y,  err)
        if string:
            return f'{mae:.4f}'
        return float(mae)

    @staticmethod
    def RMSE(one: 'DataFrame', other: 'DataFrame', string: bool=False) -> Union[str, float]:
        err = one.data - other.data
        sqr = err ** 2
        rmse = reduce(lambda x,y: x+y,  sqr) ** 0.5
        if string:
            return f'{rmse:.4f}'
        return float(rmse)
    
    @staticmethod
    def linear(one: 'DataFrame', other: 'DataFrame', string: bool=False) -> Union[str, Tuple[float, float, float]]:
        coef = np.polyfit(one.data, other.data, 1)
        k, b = coef.tolist()
        formula = np.poly1d(coef)
        pred = DataFrame(formula(one.data))
        R2 = DataFrame.R2(other, pred)
        if string:
            return f'y = {k:.4f} * x + {b:.4f}   R^2={R2:.4f}'
        return k, b, R2    # y = kx + b,    R^2=     return k, b, R2

    @staticmethod
    def pearson(one: 'DataFrame', other: 'DataFrame', string: bool=False) -> Union[str, float]:
        r = np.corrcoef(
            np.array([ one.data, other.data ])
        )[0, 1]
        r = float(r)
        if string:
            return f'{r:.4f}'
        return r
    
    @staticmethod
    def R2(real: 'DataFrame', pred: 'DataFrame') -> float:
        mean = np.mean(real.data)
        tmp = np.sum((real.data - pred.data) ** 2) / np.sum((real.data - mean) ** 2)
        return float(1 - tmp)
    
    @classmethod
    def replace_text(cls, text: str, x: 'DataFrame', y: 'DataFrame') -> str:
        cls.calc_name = ['%MAE%', '%RMSE%', '%PEARSON%', '%LINEAR%']
        cls.func: Callable[['DataFrame', 'DataFrame'], Any] = [
            cls.MAE, cls.RMSE, cls.pearson, cls.linear
        ]
        for name, func in zip(cls.calc_name, cls.func):
            if name in text:
                return text.replace(name, func(x, y, True))
        return text
        

class FileData():
    def __init__(self, file_name: str) -> None:
        self.file_name: str = file_name
        self.read_data()
        self.raw_data: List[List[str]]
        self.data: List[Optional[DataFrame]]
    
    def has_num(self, data_list: List[Any]) -> bool:
        for each in data_list:
            try:
                _ = float(each)
            except:
                pass
            else:
                return True
        return False

    def read_data(self):
        with open(self.file_name) as f:
            file_content: List[str] = f.read().splitlines()
        self.raw_data = [ x.split() for x in file_content ]
        if (not self.has_num(self.raw_data[0])) and self.has_num(self.raw_data[1]):
            self.raw_data.pop(0)
            print('drop first line')
        column = len(self.raw_data[0])
        self.data = [ None for _ in range(column) ]
        # for i in range(column):
        #     self.data.append( DataFrame([ x[i] for x in data ]) )
    
    def get(self, column: int) -> DataFrame:
        if self.data[column - 1]:
            return self.data[column - 1]
        try:
            data = DataFrame([ float(x[column - 1]) for x in self.raw_data ])
        except:
            print(f'the {column} index column data in the file "{self.file_name}" are not number')
            exit()
        self.data[column - 1] = data
        return data
        

@dataclass
class PlotElement():
    x_data: DataFrame = None
    y_data: DataFrame = None
    label: str = None
    style: Union[str] = None
    # [style] & [color, line, marker]  choose one from this two style type
    color: str = None
    line: Union[str, List[_str_num]] = None   # linestyle[, linewidth]
    marker: Union[str, List[_str_num]] = None   # marker[, markersize]
    alpha: float = None
    type: str = None  # xy or x2y


@dataclass
class ScatterElement():
    x_data: DataFrame = None
    y_data: DataFrame = None
    label: str = None
    marker: str = None
    color: Union[str, Tuple[float, float, float]] = None
    facecolor: Union[str, Tuple[float, float, float]] = None
    alpha: float = None
    linewidth: _num = None
    size: _num = None  # only s can use 1 size, sizes must convey a array-like object
    
    
@dataclass
class TextElement():
    x: float = None
    y: float = None
    content: str = None
    color: str = None
    fontsize: _num = None
    alpha: float = None
    # style: List[str, _num, float] = None  # color, fontsize, alpha
    # other: Dict[str, Any]


@dataclass
class AxesElement():
    xlabel: str = None
    ylabel: str = None
    title: str = None
    xlim: Sequence[_num] = None
    ylim: Sequence[_num] = None
    

class PlotBaseClass():
    def fig_init(self, figsize: Sequence[_num]):
        figsize = figsize if figsize else (6, 4)
        self._fig, self._ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        self._ax_x2y = None
    
    def second_y_axis(self):
        if not self._ax_x2y:
            self._ax_x2y = self._ax.twinx()
        return self._ax_x2y
    
    def legend(self):
        self._ax.legend()
        if self._ax_x2y:
            self._ax_x2y.legend()

    def eval(self, func: str, arg: List[str], key: List[str], key_expr: List[str], value: List[Any]):
        if len(key) != len(value):
            print('internal error: PlotBaseClass key and value not match')
            exit()
        kw_list = []
        for k, k_expr, v in zip(key, key_expr, value):
            if v:
                kw_list.append(f'{k}={k_expr}')
        cmd = f'{func}({", ".join(arg + kw_list)})'
        return cmd
    
    def plot(self, e: PlotElement, ax=None) -> None:
        ax = ax if ax else self._ax
        x_data = e.x_data.data
        y_data = e.y_data.data
        if e.style:
            ax.plot(x_data, y_data, e.style, label=e.label, alpha=e.alpha)
        else:
            if e.line:
                if type(e.line) == str:
                    line_style, line_width = e.line, None
                elif len(e.line) == 1:
                    line_style, line_width = e.line[0], None
                elif len(e.line) == 2:
                    line_style, line_width = e.line
                else:
                    print('error in line definition')
                    exit()
            else:
                line_style, line_width = None, None
            if e.marker:
                if type(e.marker) == str:
                    marker, marker_size = e.marker, None
                elif len(e.marker) == 1:
                    marker, marker_size = e.marker[0], None
                elif len(e.marker) == 2: 
                    marker, marker_size = e.marker
                else:
                    print('error in marker definition')
                    exit()
            else:
                marker, marker_size = None, None
            ax.plot(x_data, y_data, label=e.label, alpha=e.alpha,
                    color=e.color, linestyle=line_style, linewidth=line_width,
                    marker=marker, markersize=marker_size)
    
    def scatter(self, e: ScatterElement, ax=None) -> None:
        ax = ax if ax else self._ax
        x_data = e.x_data.data
        y_data = e.y_data.data
        ax.scatter(x_data, y_data, label=e.label, marker=e.marker,
                   color=e.color, facecolor=e.facecolor, alpha=e.alpha, 
                   linewidth=e.linewidth, s=e.size)
    
    def text(self, e: TextElement, ax=None) -> None:
        ax = ax if ax else self._ax
        color = e.color
        fontsize = e.fontsize
        alpha = e.alpha
        key = ['color', 'fontsize', 'alpha']
        value = [color, fontsize, alpha]
        eval(self.eval('ax.text', ['e.x', 'e.y', 'e.content'], key, key, value))
        # ax.text(e.x, e.y, e.content, color=color, fontsize=fontsize, alpha=alpha)

    def ax_set(self, e: AxesElement, ax=None) -> None:
        ax = ax if ax else self._ax
        if e.xlabel: ax.set_xlabel(e.xlabel)
        if e.ylabel: ax.set_ylabel(e.ylabel)
        if e.title:  ax.set_title(e.title)
        if e.xlim:   ax.set_xlim(e.xlim)
        if e.ylim:   ax.set_ylim(e.ylim)
    
    def ax_spin(self, spine: str, operator: str):
        pass

    def fig_save(self, file_name: str):
        self._fig.savefig(file_name)


class YamlPlot(PlotBaseClass):
    s = StringConst()
    task_type = [s.plot, s.scatter, s.text]
    def __init__(self, config_name: str='plot.yaml') -> None:
        self.config = self.read_yaml(config_name)
        self.fig_init()
        self.file_data: Optional[List[FileData]] = None
        # todo: analyze and plot
        self.analyze_plot()
        self.legend()
        self.fig_save()
        # plt.show()
    
    def fig_init(self):
        s = self.s
        fig = self.config.get(s.fig)
        figsize: Sequence[_num] = fig.get(s.size)
        super().fig_init(figsize)
        # axes detail define
        prop = [s.title, s.x_label, s.y_label, s.x_limit, s.y_limit]
        axes = AxesElement(
            title = fig.get(s.title),
            xlabel = fig.get(s.x_label),
            ylabel = fig.get(s.y_label),
            xlim = fig.get(s.x_limit),
            ylim = fig.get(s.y_limit)
        )
        self.ax_set(axes)

    def fig_save(self):
        s = self.s
        out_name: str = self.config.get(s.fig).get(s.out_file)
        super().fig_save(out_name)
        self.imgcat(out_name)
    
    def read_yaml(self, config_file: str) -> Dict[str, Any]:
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    
    @property
    def file_and_ope(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        s = self.s
        file_and_operation: List[Dict[str, Any]] = self.config.get(s.file)
        file_name: List[str] = [ x.get(s.file_name) for x in file_and_operation ]
        operation: List[Dict[str, Any]] = [
                { x: file_ope_dict.get(x) 
                            for x in file_ope_dict.keys() if x != s.file_name } 
                     for file_ope_dict in file_and_operation
        ]
        return file_name, operation
    
    def analyze_plot(self):
        s = self.s
        c = self.config
        file_names, operations = self.file_and_ope
        self.file_data = [ FileData(x) for x in file_names ]
        for file_idx, ope in enumerate(operations):  # for the operations of each file
            file_name = file_names[file_idx]
            for key in ope.keys():    # a list of plot operation
                if key == s.plot:
                    self.task_plot_execute(ope[key], self.file_data[file_idx])
                elif key == s.scatter:
                    self.task_scatter_execute(ope[key], self.file_data[file_idx])
                elif key == s.label:
                    pass  # todo: to be implemented. to plot multiple label at the same time (according to the file content to decide where to plot)
        texts = self.config.get(s.text)
        if texts:
            for text in texts:
                self.task_text_execute(*text)
            
    @staticmethod
    def ele(prop: Union[Sequence[_str_num], _str_num], idx: Optional[int]):
        if not prop:
            return None
        if idx:
            return prop[idx]
        else:
            return prop
    
    def check_txt_and_plot(self, x: DataFrame, y: DataFrame, task: Dict[str, Any]):
        text = task.get(self.s.text)
        if text:
            if len(text) < 3:  
                print(f'text element not complete, format: [x, y, content, [color, fontsize, alpha]] \n the previous 3 item is compulsory')
                exit()
            new_text = DataFrame.replace_text(text[2], x, y)
            try:
                style = text[3]
                self.task_text_execute(text[0], text[1], new_text, style)
            except:
                self.task_text_execute(text[0], text[1], new_text)

    def y_bias_process(self, task: Dict[str, Any], y_idx: Union[int, List[int]]) -> Optional[List[Any]]:
        bias = task.get(self.s.bias, None)
        if bias:
            return bias[1]
            # if type(y_idx) is list:
            #     return [ bias[1][i] for i in range(len(y_idx)) ]
            # else:
            #     return bias[1]
        else:
            if type(y_idx) is list:
                return [ None for _ in range(len(y_idx)) ]
            else:
                return None
        
    def task_plot_execute(self, tasks: List[Dict[str, Any]], data: FileData, ax=None):
        s = self.s
        plot_ele = [s.xy_idx, s.label, s.alpha, s.style, s.color, s.line, s.marker]
            
        def plot_one(x: DataFrame, y: DataFrame, task: Dict[str, List[Any]]):
            label = task.get(s.label)
            style = task.get(s.style)
            line = task.get(s.line)
            color = task.get(s.color)
            marker = task.get(s.marker)
            alpha = task.get(s.alpha)
            if style:
                e = PlotElement(x_data=x, y_data=y, label=label, alpha=alpha, style=style)
            else:
                e = PlotElement(x_data=x, y_data=y, label=label, alpha=alpha,
                                line=line, color=color, marker=marker)
            if task.get(s.type) == s.x2y:
                axes = self.second_y_axis()
            else:
                axes = ax
            self.plot(e, axes)
            self.check_txt_and_plot(x, y, task)

        for each_task in tasks:
            x, y = each_task[s.xy_idx]  # may cause exception
            bias = each_task.get(s.bias, None)
            x_data = data.get(x) + bias[0] if bias else data.get(x)
            y_bias = self.y_bias_process(each_task, y)
            if type(y) == list:   # multiple plot
                for idx, sep_task in enumerate(self.separate_task(each_task)):
                    y_data = data.get(y[idx]) + y_bias[idx]
                    # plot_one(data.get(x), data.get(y[idx]), sep_task)
                    plot_one(x_data, y_data, sep_task)

                # for idx, one_y in enumerate(y):
                #     plot_one(data.get(x), data.get(one_y), each_task, idx)
            else:  # single plot
                y_data = data.get(y) + y_bias
                # plot_one(data.get(x), data.get(y), each_task)
                plot_one(x_data, y_data, each_task)


    def task_scatter_execute(self, tasks: List[Dict[str, Any]], data: FileData, ax=None):
        s = self.s
        scatter_ele = [s.xy_idx, s.label, s.marker, s.alpha, s.color, s.facecolor, s.linewidth, s.size]
        # def ele(prop: Union[List[_str_num], _str_num], idx: Optional[int]) -> _str_num:
            
        def plot_one(x: DataFrame, y: DataFrame, task: Dict[str, List[Any]]):
            label = task.get(s.label)
            marker = task.get(s.marker)
            color = task.get(s.color)
            facecolor = task.get(s.facecolor)
            alpha = task.get(s.alpha)
            linewidth = task.get(s.linewidth)
            size = task.get(s.size)
            e = ScatterElement(x_data=x, y_data=y, label=label, marker=marker,
                               color=color, facecolor=facecolor, alpha=alpha,
                               linewidth=linewidth, size=size)
            if task.get(s.type) == s.x2y:
                axes = self.second_y_axis()
            else:
                axes = ax
            self.scatter(e, axes)
            self.check_txt_and_plot(x, y, task)

        for each_task in tasks:
            x, y = each_task[s.xy_idx]  # may cause exception
            bias = each_task.get(s.bias, None)
            x_data = data.get(x) + bias[0] if bias else data.get(x)
            y_bias = self.y_bias_process(each_task, y)
            if type(y) == list:   # multiple plot
                for idx, sep_task in enumerate(self.separate_task(each_task)):
                    y_data = data.get(y[idx]) + y_bias[idx]
                    # plot_one(data.get(x), data.get(y[idx]), sep_task)
                    plot_one(x_data, y_data, sep_task)
                # for idx, one_y in enumerate(y):
                #     plot_one(data.get(x), data.get(one_y), each_task, idx)
            else:  # single plot
                y_data = data.get(y) + y_bias
                # plot_one(data.get(x), data.get(y), each_task)
                plot_one(x_data, y_data, each_task)
    
    def task_text_execute(self, x: _num, y: _num, content: str, style: Sequence[_str_num]=None, ax=None):
        e = TextElement(x=x, y=y, content=content)
        try:
            e.color = style[0]
            e.fontsize = style[1]
            e.alpha = style[2]
        except:
            pass
        self.text(e, ax)
    
    @classmethod
    def separate_task(cls, tasks: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        s = cls.s
        xys = tasks[s.xy_idx]  # [x, [y1, y2, y3, y4]]
        task_num = len(xys[1])
        new_task = [ {} for _ in range(task_num) ]
        for idx, xy in enumerate(xys[1]):
            new_task[idx][s.xy_idx] = [xys[0], xys[1][idx]]
            for key in tasks.keys():
                if key == s.xy_idx:
                    continue
                elif key == s.bias:
                    continue
                new_task[idx][key] = tasks.get(key)[idx]
        return new_task

    def imgcat(self, out_name: str):
        os.system(f'imgcat {out_name}')
        

if __name__ == "__main__":
    # y = YamlPlot('test.yaml')
    try:
        yaml_argv = sys.argv[1]
        y = YamlPlot(yaml_argv)
    except:
        yaml_files = list(filter(lambda x: x.endswith('.yaml') or x.endswith('.yml'), os.listdir()))
        if len(yaml_files) == 1:
            y = YamlPlot(yaml_files[0])
        elif len(yaml_files) == 0:
            print('no yaml file exists in current directory!')
        else:
            print('no specific yaml file in argument and multiple yaml files exists at current directory')

