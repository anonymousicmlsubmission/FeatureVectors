import os
import sys
import numpy as np
import _pickle as pkl
import matplotlib.pyplot as plt
import plotly.express as px
from rulefit import RuleFit
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
from colorsys import hsv_to_rgb
from tqdm import tqdm
from sklearn.tree import _tree
from utils import *
from scipy.stats import random_correlation
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import plotly.graph_objects as go

class FeatureVec(object):
    "Feature-vector class."
    def __init__(
        self, mode, max_depth=3, feature_names=None, max_sentences=20000,
        exp_rand_tree_size=True, tree_generator=None,
    ):
        '''
        mode: 'classify' or 'regress'
        max_depth: maximum depth of trained trees
        feature_names: names of features
        max_sentences: maximum number of extracted sentences
        exp_rand_tree_size: Having trees with different sizes
        tree_generator: Tree generator model (overwrites above features)
        '''
        self.feature_names = feature_names
        self.mode = mode
        max_leafs = 2 ** max_depth
        num_trees = max_sentences // max_leafs
        if tree_generator is None:
            tree_generator = RandomForestClassifier(num_trees, max_depth=max_depth)
        self.exp_rand_tree_size = exp_rand_tree_size
        self.rf = RuleFit(
            rfmode=mode, tree_size=max_leafs, max_rules=max_sentences, tree_generator=tree_generator,
            exp_rand_tree_size=True, fit_lasso=False, Cs=10.**np.arange(-4, 1), cv=3)
        
    def fit(self, X, y, restart=True, bagging=0):
        '''Fit the tree model.
        X: inputs
        y: outputs (integer class label or real value)
        restart: To train from scratch tree generator model
        bagging: If >0 applies bagging on trees to compute confidence intervals
        '''
        
        
        if not bagging:
            bagging = 0
        
        dimred = TruncatedSVD(2)
        self.rf.fit(X, y, restart=restart)
        rules = self.rf.get_rules()['rule'].values
        cm = cooccurance_matrix(rules, X.shape[-1])
        vectors = dimred.fit_transform(cm)
        vectors = normalize_angles(vectors)
        self.norms = np.clip(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12, None)
        vectors /= np.max(self.norms)
        self.vectors = vectors
        self.importance = np.linalg.norm(self.vectors, axis=-1)
        self.angles = np.arctan2(self.vectors[:, 1], self.vectors[:, 0])
        self.stds = np.zeros(vectors.shape)
        self.predictor = self.rf.tree_generator
        if bagging:
            all_vectors = []
            for _ in range(bagging):
                self.rf.bag_trees(X, y)
                rules_bag = self.rf.get_rules()['rule'].values
                cm_bag = cooccurance_matrix(rules_bag, X.shape[-1])
                vectors_bag = dimred.fit_transform(cm_bag)
                vectors_bag = normalize_angles(vectors_bag)
                norms_bag = np.clip(np.linalg.norm(vectors_bag, axis=-1, keepdims=True), 1e-12, None)
                all_vectors.append(vectors_bag / norms_bag)
            self.stds = np.std(all_vectors, 0)

    def plot(self, dynamic=True, confidence=True, path=None):
        '''Plot the feature-vectors.
        dynamic: If True the output is a dynamic html plot. Otherwise, it will be an image.
        confidence: To show confidence intervals or not
        path: Path to save the image. If dy
        '''
        mx = 1.1
        angles = np.arctan2(self.vectors[:, 1], self.vectors[:, 0])
        max_angle = np.max(np.abs(angles))
        feature_names = self.feature_names + ['origin', '']
        plot_vectors = np.concatenate([self.vectors, [[0, 0], [0, 0]]])
        vectors_sizes = np.linalg.norm(plot_vectors, axis=-1)
        plot_angles = np.concatenate([angles, [-max_angle, max_angle]])
        plot_data = np.stack([plot_vectors[:, 1], plot_vectors[:, 0], plot_angles, feature_names], axis=-1)
        plot_df = pd.DataFrame(
            data=plot_data,
        columns=['x', 'y', 'angles', 'names'])
        plot_df[["x", "y", "angles"]] = plot_df[["x", "y", "angles"]].apply(pd.to_numeric)
        if dynamic:
            fig = px.scatter(
                plot_df, x='x', y='y', color='angles', width=1000, height=500,
                hover_name=feature_names,
                hover_data={'x': False, 'y': False, 'angles':False, 'names':False},
                color_continuous_scale=px.colors.sequential.Rainbow)
            
            fig.update_yaxes(visible=False, showticklabels=False, range=[0, mx])
            fig.update_xaxes(visible=False, showticklabels=False, range=[-mx, mx])
        else:
            fig = px.scatter(
                plot_df, x='x', y='y', color='angles', width=1000, height=500,
                hover_name='names',
                hover_data={'x': False, 'y': False, 'angles':False, 'names':False},
                color_continuous_scale=px.colors.sequential.Rainbow)
            max_name_len = max([len(i) for i in feature_names])
            for i in range(len(plot_vectors) - 2):
                if plot_vectors[:, 1][i] > 0:
                    name = feature_names[i] + ''.join([' '] * (max_name_len - len(feature_names[i])))
                    ax = plot_vectors[:, 1][i] + 0.2
                else:
                    name = ''.join([' '] * (max_name_len - len(feature_names[i]))) + feature_names[i]
                    ax = plot_vectors[:, 1][i] - 0.2
                if vectors_sizes[i] < 0.2:
                    continue
                fig.add_annotation(
                    x=plot_vectors[:, 1][i],
                    y=plot_vectors[:, 0][i],
                    text=feature_names[i] + ''.join([' '] * (max_name_len - len(feature_names[i]))),
                    font=dict(size=15),
                    axref="x",
                    ayref="y",
                    ax=ax,
                    ay=plot_vectors[:, 0][i],
                    arrowhead=2,
                    )
                fig.update_yaxes(visible=False, showticklabels=False, range=[0, mx])
                fig.update_xaxes(visible=False, showticklabels=False, range=[-mx, mx])
        fig.update_traces(marker=dict(size=10), textfont_size=15)
        fig.update(layout_coloraxis_showscale=False)
        fig.update_layout(showlegend=False)
        for i in range(10):
            fig.add_shape(
                type='circle', x0=(i+1) / 10 * mx, y0=(i + 1) / 10 * mx, x1=-(i + 1) / 10 * mx, y1=-(i + 1) / 10 * mx,
                line_color="red", opacity=0.5,   line=dict(dash='dot', width=3))
        if confidence:
            for vector, std, angle in zip(self.vectors, self.stds, angles):
                fig.add_shape(
                    type='circle', x0=vector[1]+3*std[1], y0=vector[0]+3*std[0], x1=vector[1]-3*std[1], y1=vector[0]-3*std[0],
                    line_color='gray', opacity=0.5,   line=dict(dash='solid', width=1))
        fig.show()
        if path:
            if len(path.split('/')) > 1 and not os.path.exists('/'.join(path.split('/')[:-1])):
                os.makedirs('/'.join(path.split('/')[:-1]))
            if dynamic:
                assert path.split('.')[-1] == 'html', 'For a dynamic figure, path should be an html file!'
                fig.write_html(path)
            else:
                fig.write_image(path)
                
                
class KN_FeatureVec(FeatureVec):
    
    def __init__(self, mode, max_depth=3, feature_names=None, max_rules=20000,
        exp_rand_tree_size=True, Cs=None, cv=None, tree_generator=None):
        super().__init__(mode, max_depth=3, feature_names=None, max_rules=20000,
        exp_rand_tree_size=True, Cs=None, cv=None, tree_generator=None)
        if Cs is None:
            Cs = 10.**np.arange(-4, 1)
        if cv is None:
            cv = 3
        self.feature_names = feature_names
        self.mode = mode
        max_leafs = 2 ** max_depth
        num_trees = max_rules // max_leafs
        if tree_generator is None:
            tree_generator = RandomForestClassifier(num_trees, max_depth=max_depth)
        self.exp_rand_tree_size = exp_rand_tree_size
        self.rf = RuleFit(
            rfmode=mode, tree_size=max_leafs, max_rules=max_rules, tree_generator=tree_generator,
            exp_rand_tree_size=True, fit_lasso=False, Cs=Cs, cv=cv)

    def plot(self, dynamic=True, confidence=True, path=None):
        
        dim = len(self.vectors) // 2
        mx = 1.1
        
        
        angles = np.arctan2(self.vectors[:, 1], self.vectors[:, 0])
        max_angle = np.max(np.abs(angles))
#         max_angle = 1
        feature_names = self.feature_names[:dim] + ['origin', '']
        plot_vectors = np.concatenate([self.vectors[:dim], [[0, 0], [0, 0]]])
        vectors_sizes = np.linalg.norm(plot_vectors, axis=-1)
        plot_angles = np.concatenate([angles[:dim], [-max_angle, max_angle]])
        plot_data = np.stack([plot_vectors[:, 1], plot_vectors[:, 0], plot_angles, feature_names], axis=-1)
        plot_df = pd.DataFrame(
            data=plot_data,
        columns=['x', 'y', 'angles', 'names'])
        plot_df[["x", "y", "angles"]] = plot_df[["x", "y", "angles"]].apply(pd.to_numeric)
        
        feature_names_kn = self.feature_names[dim:] + ['origin', '']
        plot_vectors_kn = np.concatenate([self.vectors[dim:], [[0, 0], [0, 0]]])
        vectors_sizes_kn = np.linalg.norm(plot_vectors_kn, axis=-1)
        plot_data_kn = np.stack([plot_vectors_kn[:, 1], plot_vectors_kn[:, 0], plot_angles, feature_names_kn], axis=-1)
        plot_df_kn = pd.DataFrame(
            data=plot_data_kn,
        columns=['x', 'y', 'angles', 'names'])
        plot_df_kn[["x", "y", "angles"]] = plot_df_kn[["x", "y", "angles"]].apply(pd.to_numeric)
        if dynamic:
            fig = px.scatter(
                plot_df, x='x', y='y', color='angles', width=1000, height=500,
                hover_name=feature_names,
                hover_data={'x': False, 'y': False, 'angles':False, 'names':False},
                color_continuous_scale=px.colors.sequential.Rainbow)
            fig_kn = px.scatter(
                plot_df_kn, x='x', y='y', color='angles', width=1000, height=500,
                hover_name=feature_names_kn,
                hover_data={'x': False, 'y': False, 'angles':False, 'names':False},
                color_continuous_scale=px.colors.sequential.Rainbow, opacity=0.5)
            fig.add_trace(fig_kn.data[0])

            fig.update_yaxes(visible=False, showticklabels=False, range=[0, mx])
            fig.update_xaxes(visible=False, showticklabels=False, range=[-mx, mx])
        else:
            fig = px.scatter(
                plot_df, x='x', y='y', color='angles', width=1000, height=500,
                hover_name='names',
                hover_data={'x': False, 'y': False, 'angles':False, 'names':False},
#                 text='names',
                color_continuous_scale=px.colors.sequential.Rainbow)
            max_name_len = max([len(i) for i in feature_names])
            for i in range(len(plot_vectors) - 2):
                if plot_vectors[:, 1][i] > 0:
                    name = feature_names[i] + ''.join([' '] * (max_name_len - len(feature_names[i])))
                    ax = plot_vectors[:, 1][i] + 0.2
                else:
                    name = ''.join([' '] * (max_name_len - len(feature_names[i]))) + feature_names[i]
                    ax = plot_vectors[:, 1][i] - 0.2
                if vectors_sizes[i] < 0.2:
                    continue
                fig.add_annotation(
                    x=plot_vectors[:, 1][i],
                    y=plot_vectors[:, 0][i],
                    text=feature_names[i] + ''.join([' '] * (max_name_len - len(feature_names[i]))),
                    font=dict(size=15),
                    axref="x",
                    ayref="y",
                    ax=ax,
                    ay=plot_vectors[:, 0][i],
                    arrowhead=2,
                    )
                fig.update_yaxes(visible=False, showticklabels=False, range=[0, mx])
                fig.update_xaxes(visible=False, showticklabels=False, range=[-mx, mx])
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')), textfont_size=15)
        fig.update(layout_coloraxis_showscale=False)
        fig.update_layout(showlegend=False)
        for i in range(10):
            fig.add_shape(
                type='circle', x0=(i+1) / 10 * mx, y0=(i + 1) / 10 * mx, x1=-(i + 1) / 10 * mx, y1=-(i + 1) / 10 * mx,
                line_color="red", opacity=0.5,   line=dict(dash='dot', width=3))
        if confidence:
            for vector, std, angle in zip(self.vectors, self.stds, angles):
                fig.add_shape(
                    type='circle', x0=vector[1]+3*std[1], y0=vector[0]+3*std[0], x1=vector[1]-3*std[1], y1=vector[0]-3*std[0],
                    line_color='gray', opacity=0.5,   line=dict(dash='solid', width=1))
        fig.show()
        if path:
            if len(path.split('/')) > 1 and not os.path.exists('/'.join(path.split('/')[:-1])):
                os.makedirs('/'.join(path.split('/')[:-1]))
            if dynamic:
                assert path.split('.')[-1] == 'html', 'For a dynamic figure, path should be an html file!'
                fig.write_html(path)
            else:
                fig.write_image(path)