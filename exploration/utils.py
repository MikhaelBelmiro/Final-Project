import os
import re
import csv
import time
import torch
import pickle
import requests
import numpy as np
import pandas as pd

from torch import nn

def s2f(arr, dtype):
    new_array = np.zeros_like(arr, dtype=dtype)
    for i, deg in enumerate(arr):
        numbers = re.findall(r"\d+", str(deg))
        
        new_entry = float(numbers[0])
        new_array[i] = new_entry
    return new_array

def get_total_visits(df):
    date_columns = [col for col in df.columns if '-' in col]
    df['Total'] = 0
    for col in date_columns:
        df['Total'] += df[col]
    return df

def get_wiki_parameters(parameter_str):
    parameter_list = parameter_str.split('_')
    agent = parameter_list[-1]
    access = parameter_list[-2]
    project = parameter_list[-3]
    name = '_'.join(parameter_list[:-3])

    return {
        'path': parameter_str,
        'name':name,
        'project': project,
        'access': access,
        'agent': agent
    }

def requests_wikimedia_api(parameter):
    granularity = "daily"
    start = '20170910'
    end = '20220101'
    path = parameter['path']
    project = parameter['project']
    access = parameter['access']
    agent = parameter['agent']
    article = parameter['name']

    response = requests.get(
        f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}', 
        headers={'User-Agent': 'belmiro533backup@gmail.com'}
    )
    try:
        response = {
            'path': path,
            'response': response.json()['items']
        }
    except:
        response = {
            'path': path,
            'response': response.status_code
        }
    return response

def format_timestamp(timestamp):
    out_timestamp = f'{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}'
    return out_timestamp

def get_visits_from_responses(responses):
    out_dict = {}
    for response in responses:
        if not(isinstance(response['response'], int)):
            page = response['path']
            out_dict[page] = {}
            for daily_response in response['response']:
                daily_timestamp = format_timestamp(daily_response['timestamp'])
                daily_views = daily_response['views']
                out_dict[page][daily_timestamp] = daily_views
    return out_dict

def read_from_csv_custom(file_path, columns, columns_dtypes='auto', delimiter='|', quotechar='"', escapechar='\\', encoding='utf-8', return_pd=True):
    df = {col:[] for col in columns}
     
    with open(file_path, mode='rb') as file:
        for i, line in enumerate(file):
            try:
                line = line.decode(encoding)
                csv_reader = csv.reader([line], delimiter=delimiter, quotechar=quotechar, escapechar=escapechar)
                for csv_line in csv_reader:
                    for i, col in enumerate(columns):
                        df[col].append(csv_line[i])
            except:
                continue
    
    if return_pd:
        if columns_dtypes == 'auto':
            return pd.DataFrame(df)
        else:
            pd.DataFrame(df, dtype=columns_dtypes)

    else:
        return df

class Graph:
    def __init__(self):
        pass

    def iterate_from_file(self, filename, redirects):
        self.nodes = set()
        self.adjacency_list = {}

        with open(filename, mode='rb') as file:
            i = 0
            for line in file:
                try:
                    line = line.decode('utf-8')
                    csv_reader = csv.reader([line], delimiter='|', quotechar='"', escapechar='\\')
                    for csv_line in csv_reader:                        
                        pl_from, pl_to = int(csv_line[0]), int(csv_line[1])

                        try:
                            pl_to = redirects[pl_to]['rd_title']
                        except:
                            pass

                        try:
                            pl_from = redirects[pl_from]['rd_title']
                        except:
                            pass
                        
                        self.nodes.update([pl_from, pl_to])
                        try:
                            self.adjacency_list[pl_from].add((pl_to, 1.0))
                        except:
                            self.adjacency_list[pl_from] = set()
                            self.adjacency_list[pl_from].add((pl_to, 1.0))
                except:
                    continue
                i += 1
    
    def read_from_file(self, node_list_filename, adjacency_list_filename):
        with open(node_list_filename, 'rb') as node_list:
            self.nodes = pickle.load(node_list)

        with open(adjacency_list_filename, 'rb') as adj_list:
            self.adjacency_list = pickle.load(adj_list)


    def shortest_path(self, start_node, end_node=None):
        if end_node:
            if not(isinstance(end_node, list)):
                end_node = [end_node]
            end_node_copy = end_node.copy()

        unvisited_nodes = self.nodes.copy()

        distance_from_start = {
            node: (0 if node == start_node else float("inf")) for node in self.nodes
        }

        i = 0
        while unvisited_nodes:
            current_node = min(
                unvisited_nodes, key=lambda node: distance_from_start[node]
            )
            unvisited_nodes.remove(current_node)
            if current_node in end_node:
                end_node_copy.remove(current_node)

            if distance_from_start[current_node] == float("inf"):
                break

            for neighbor, distance in self.adjacency_list[current_node]:
                new_path = distance_from_start[current_node] + distance
                if new_path < distance_from_start[neighbor]:
                    distance_from_start[neighbor] = new_path

            if end_node is not None and len(end_node_copy)==0:
                break
            i+=1
    
        if end_node is not None:
            return {node:distance_from_start[node] for node in end_node}
        else:
            return distance_from_start

def create_adj_from_mapping(mapper, adj_mat_dict, randomizer=True):
    n = len(mapper)
    inverse_mapper = {val[1]:key for key, val in mapper.items()}

    adj_mat = np.zeros(shape=(n, n))
    for row_id in adj_mat_dict:
        row_title = inverse_mapper[row_id]
        row_index, _ = mapper[row_title]
        
        col_dict = dict(adj_mat_dict[row_id])
        col_entries = np.zeros(shape=n)

        for col_id, entry in col_dict.items():
            if randomizer:
                rand = np.random.uniform(0, 0.4)
            else:
                rand = 0

            col_title = inverse_mapper[col_id]
            col_index, _ = mapper[col_title]

            col_entries[col_index] = entry+rand

        adj_mat[row_index, :] = col_entries
        adj_mat[row_index, row_index] = 0
    adj_mat = adj_mat + adj_mat.T
    np.fill_diagonal(adj_mat, 1e-4)
    return adj_mat