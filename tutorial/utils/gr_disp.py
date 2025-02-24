"""
Utility for displaying graph with Tensorboard in Jupyter notebooks in tf2
""" 

import sys
import os
import shutil

import tensorflow.compat.v1 as tf


def _load_graph_def(file_path):
	with tf.gfile.GFile(file_path, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def_str = f.read()
		graph_def.ParseFromString(graph_def_str)
	return graph_def

def _save_graph_for_tb():
	gd = _load_graph_def('logs/graph.pb')
	g=tf.Graph()
	with g.as_default():
		tf.import_graph_def(gd, name="")
	
	summary_writer = tf.summary.FileWriter('./logs', graph=g)
  
def show_graph(g=None, gd=None):
	if os.path.exists('logs'):
		shutil.rmtree('logs')
		
	if gd and g is None:
		g=tf.compat.v1.Graph()
		with g.as_default():
			tf.compat.v1.import_graph_def(gd, name="")
	
	if g is None:
		g = tf.get_default_graph()
			
	summary_writer = tf.summary.FileWriter('./logs', graph=g)

def show_graph_eager(g=None, gd=None):
	if g is None and gd is None:
		g = tf.compat.v1.get_default_graph()
	
	if g and gd is None:
		gd = g.as_graph_def()
	
	if os.path.exists('logs'):
		shutil.rmtree('logs')
	os.makedirs('logs')
	
	with tf.compat.v1.gfile.GFile('logs/graph.pb', "wb") as f:
		f.write(gd.SerializeToString())
		
	os.system('python utils/gr_disp.py')
	
if __name__ == "__main__":
	tf.disable_v2_behavior()
	_save_graph_for_tb()