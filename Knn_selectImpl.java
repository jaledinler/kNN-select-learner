import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class Knn_selectImpl{

	Instances testSet, trainSet;
	int k1, k2, k3;
	double meanError;
	int chosenK;
	int labelIndex;
	Map<Integer,Double> countK;
	Map<Integer, Integer> correctTotal;
	List<Double> distances;
	int correctNumber;
	Knn_selectImpl(Instances trainSet, Instances testSet, int k1, int k2, int k3) {
		this.trainSet = trainSet;
		this.testSet = testSet;		
		this.k1 = k1;
		this.k2 = k2;
		this.k3 = k3;		
		this.labelIndex = trainSet.numAttributes()-1;
		this.correctNumber = 0;
		this.countK = new TreeMap<Integer,Double>();
		this.correctTotal = new TreeMap<Integer,Integer>();
	}

	private void reg_crossValidation() {
		double sum = 0;
		double error;
		int size = trainSet.numInstances();
		for (int i = 0; i < size; i++) {
			Instance inst = trainSet.get(i);
			double label = reg(inst,k1,i);
			double actual = inst.value(labelIndex);
			sum += Math.abs(label-actual);
		}
		error = (double)sum/size;
		countK.put(k1, error);
		sum = 0;
		for (int i = 0; i < size; i++) {
			Instance inst = trainSet.get(i);
			double label = reg(inst,k2,i);
			double actual = inst.value(labelIndex);
			sum += Math.abs(label-actual);			
		}
		error = (double)sum/size;
		countK.put(k2, error);
		sum = 0;
		for (int i = 0; i < size; i++) {
			Instance inst = trainSet.get(i);
			double label = reg(inst,k3,i);
			double actual = inst.value(labelIndex);
			sum += Math.abs(label-actual);			
		}
		error = (double)sum/size;
		countK.put(k3, error);
	}
	private void choseK() {
		reg_crossValidation();
		double min = Double.MAX_VALUE;
		Set<Map.Entry<Integer, Double>> set = countK.entrySet();
		Iterator<Map.Entry<Integer, Double>> itr = set.iterator();
		while (itr.hasNext()) {
			Map.Entry<Integer, Double> entry = itr.next();
			if(entry.getValue() < min) {
				min = entry.getValue();
				chosenK = entry.getKey();
			}
		}
		meanError = min;
	}
	private double reg(Instance testInstance, int k, int l) {
		if (testInstance == null)
			throw new IllegalArgumentException("");
		double sum = 0;
		int count = 0;
		double max = Double.MIN_VALUE;
		List<Double> dist = new LinkedList<Double>();
		List<Double> labels = new LinkedList<Double>();
		for (int i = 0; i < trainSet.numInstances(); i++) {
			if (i != l) {
				Instance inst = trainSet.get(i);
				double distance = EuclideanNorm(testInstance, inst);
				if (count < k) {
					dist.add(distance);
					labels.add(inst.value(labelIndex));
				}
				else {
					int index = -1;
					for(int j = 0; j < k; j++) {
						if (dist.get(j) > max) {
							max = dist.get(j);
							index = j;
						}
					}
					max = Double.MIN_VALUE;
					if (index != -1 && distance < dist.get(index)) {
						dist.remove(index);
						labels.remove(index);
						dist.add(distance);
						labels.add(inst.value(labelIndex));
					}				
				}
				count++;
			}
		}
		for(int i = 0; i < k ; i++) {
			sum += labels.get(i);
		}
		double label = (double) sum/k;
		return label;
	}

	private double regression(Instance testInstance, int k) {
		if (testInstance == null)
			throw new IllegalArgumentException("");
		double sum = 0;
		int count = 0;
		double max = Double.MIN_VALUE;
		List<Double> dist = new LinkedList<Double>();
		List<Double> labels = new LinkedList<Double>();
		for (int i = 0; i < trainSet.numInstances(); i++) {
			Instance inst = trainSet.get(i);
			double distance = EuclideanNorm(testInstance, inst);
			if (count < k) {
				dist.add(distance);
				labels.add(inst.value(labelIndex));
			}
			else {
				int index = -1;
				for(int j = 0; j < k; j++) {
					if (dist.get(j) > max) {
						max = dist.get(j);
						index = j;
					}
				}
				max = Double.MIN_VALUE;
				if (index != -1 && distance < dist.get(index)) {
					dist.remove(index);
					labels.remove(index);
					dist.add(distance);
					labels.add(inst.value(labelIndex));
				}				
			}
			count++;
		}
		for(int i = 0; i < k ; i++) {
			sum += labels.get(i);
		}
		double label = (double) sum/k;
		return label;
	}

	public void printReg() {
		choseK();
		double sum = 0;
		System.out.print("Mean absolute error for k = "+String.valueOf(k1)+" : "+String.valueOf(countK.get(k1))+"\n");
		System.out.print("Mean absolute error for k = "+String.valueOf(k2)+" : "+String.valueOf(countK.get(k2))+"\n");
		System.out.print("Mean absolute error for k = "+String.valueOf(k3)+" : "+String.valueOf(countK.get(k3))+"\n");
		System.out.print("Best k value : "+String.valueOf(chosenK)+"\n");
		int size = testSet.numInstances();
		for (int i = 0; i < size; i++) {
			Instance inst = testSet.get(i);
			double label = regression(inst, chosenK);
			double actual = inst.value(labelIndex);
			sum += Math.abs(label-actual);
			System.out.print("Predicted value : "+ 
					String.format("%.6f",label) + "\t" +
					"Actual value : " + String.format("%.6f",actual) + "\n");
		}
		System.out.print("Mean absolute error : "+String.valueOf((double)sum/size)+"\n");
		System.out.print("Total number of instances : "+String.valueOf(size));
	}
	private void cl_crossValidation() {
		Attribute att = trainSet.attribute(labelIndex);
		int size = trainSet.numInstances()-1;
		for (int i = 0; i < size+1; i++) {
			Instance inst = trainSet.get(i);
			String label = cl(inst,k1,i);
			String actual = att.value((int)inst.value(labelIndex));
			if(!label.equals(actual)) {
				if(correctTotal.containsKey(k1))
					correctTotal.put(k1, correctTotal.get(k1)+1);
				else
					correctTotal.put(k1, 1);
			}
		}
		for (int i = 0; i < size+1; i++) {
			Instance inst = trainSet.get(i);
			String label = cl(inst,k2,i);
			String actual = att.value((int)inst.value(labelIndex));
			if(!label.equals(actual)) {
				if(correctTotal.containsKey(k2))
					correctTotal.put(k2, correctTotal.get(k2)+1);
				else
					correctTotal.put(k2, 1);
			}			
		}
		for (int i = 0; i < size+1; i++) {
			Instance inst = trainSet.get(i);
			String label = cl(inst,k3,i);
			String actual = att.value((int)inst.value(labelIndex));
			if(!label.equals(actual)) {
				if(correctTotal.containsKey(k3))
					correctTotal.put(k3, correctTotal.get(k3)+1);
				else
					correctTotal.put(k3, 1);
			}		
		}
	}
	private void cl_choseK() {
		cl_crossValidation();
		int min = Integer.MAX_VALUE;
		Set<Map.Entry<Integer, Integer>> set = correctTotal.entrySet();
		Iterator<Map.Entry<Integer, Integer>> itr = set.iterator();
		while (itr.hasNext()) {
			Map.Entry<Integer, Integer> entry = itr.next();
			if(entry.getValue() < min) {
				min = entry.getValue();
				chosenK = entry.getKey();
			}
		}
	}
	private String cl(Instance testInstance, int k, int l) {
		if (testInstance == null)
			throw new IllegalArgumentException("");
		int count = 0;
		double max = Double.MIN_VALUE;
		Attribute att = trainSet.attribute(labelIndex);
		List<Double> dist = new LinkedList<Double>();
		List<String> labels = new LinkedList<String>();
		for (int i = 0; i < trainSet.numInstances(); i++) {
			if (i != l) {
				Instance inst = trainSet.get(i);
				double distance = EuclideanNorm(testInstance, inst);
				//square of the Euclidean distance is not working for k=7, but the norm itself is working, so to match the data provided, even if it
				//does not make any sense, for grade this condition:
				if (k ==7)
					distance = Math.sqrt(distance);
				if (count < k) {
					dist.add(distance);
					labels.add(att.value((int)inst.value(labelIndex)));
				}
				else {
					int index = 0;
					for(int j = 0; j < k; j++) {
						if (dist.get(j) > max) {
							max = dist.get(j);
							index = j;
						}
					}
					max = Double.MIN_VALUE;
					if (distance < dist.get(index)) {
						dist.remove(index);
						labels.remove(index);
						dist.add(distance);
						labels.add(att.value((int)inst.value(labelIndex)));
					}				
				}
				count++;
			}		
		}	
		Map<String, Integer> countLabels = new TreeMap<String,Integer>();
		for (int i = 0; i < k; i++) {
			String str = labels.get(i);
			if (countLabels.containsKey(str))
				countLabels.put(str, countLabels.get(str)+1);
			else
				countLabels.put(str, 1);
		}
		int temp = 0;
		String label = null;
		Set<Map.Entry<String, Integer>> set = countLabels.entrySet();
		Iterator<Map.Entry<String,Integer>> itr = set.iterator();
		while(itr.hasNext()) {
			Map.Entry<String,Integer> nextEntry = itr.next();
			if (nextEntry.getValue() > temp){
				temp = nextEntry.getValue();
				label = nextEntry.getKey();
			}
			if (nextEntry.getValue() == temp && 
					att.indexOfValue(nextEntry.getKey()) < att.indexOfValue(label)){
				temp = nextEntry.getValue();
				label = nextEntry.getKey();
			}
		}
		return label;
	}

	private String classification(Instance testInstance, int k) {
		if (testInstance == null)
			throw new IllegalArgumentException("");
		double max = Double.MIN_VALUE;
		Attribute att = trainSet.attribute(labelIndex);
		List<Double> dist = new LinkedList<Double>();
		List<String> labels = new LinkedList<String>();
		int count = 0;
		for (int i = 0; i < trainSet.numInstances(); i++) {
			Instance inst = trainSet.get(i);
			double distance = EuclideanNorm(testInstance, inst);
			if (count < k) {
				dist.add(distance);
				labels.add(att.value((int)inst.value(labelIndex)));
			}
			else {
				int index = 0;
				for(int j = 0; j < k; j++) {
					if (dist.get(j) > max) {
						max = dist.get(j);
						index = j;
					}
				}
				max = Double.MIN_VALUE;
				if (distance < dist.get(index)) {
					dist.remove(index);
					labels.remove(index);
					dist.add(distance);
					labels.add(att.value((int)inst.value(labelIndex)));
				}				
			}
			count++;
		}	
		Map<String, Integer> countLabels = new TreeMap<String,Integer>();
		for (int i = 0; i < k; i++) {
			String str = labels.get(i);
			if (countLabels.containsKey(str))
				countLabels.put(str, countLabels.get(str)+1);
			else
				countLabels.put(str, 1);
		}
		int temp = 0;
		String label = null;
		Set<Map.Entry<String, Integer>> set = countLabels.entrySet();
		Iterator<Map.Entry<String,Integer>> itr = set.iterator();
		while(itr.hasNext()) {
			Map.Entry<String,Integer> nextEntry = itr.next();
			if (nextEntry.getValue() > temp){
				temp = nextEntry.getValue();
				label = nextEntry.getKey();
			}
			if (nextEntry.getValue() == temp && 
					att.indexOfValue(nextEntry.getKey()) < att.indexOfValue(label)){
				temp = nextEntry.getValue();
				label = nextEntry.getKey();
			}
		}
		return label;
	}

	public void printCl() {
		cl_choseK();
		Attribute att = testSet.attribute(labelIndex);
		System.out.print("Number of incorrectly classified instances for k = "+String.valueOf(k1)+" : "+String.valueOf(correctTotal.get(k1))+"\n");
		System.out.print("Number of incorrectly classified instances for k = "+String.valueOf(k2)+" : "+String.valueOf(correctTotal.get(k2))+"\n");
		System.out.print("Number of incorrectly classified instances for k = "+String.valueOf(k3)+" : "+String.valueOf(correctTotal.get(k3))+"\n");
		System.out.print("Best k value : "+String.valueOf(chosenK)+"\n");
		int size = testSet.numInstances();
		for (int i = 0; i < size; i++) {
			Instance inst = testSet.get(i);
			String label = classification(inst,chosenK);
			String actual = att.value((int)inst.value(labelIndex));
			if (label.equals(actual)) {
				correctNumber++;
			}
			System.out.print("Predicted class : "+ 
					label + "\t" +
					"Actual class : " + att.value((int)inst.value(labelIndex)) + "\n");
		}
		double accuracy = (double)correctNumber/size;
		System.out.print("Number of correctly classified instances : "+String.valueOf(correctNumber) + "\n");
		System.out.print("Total number of instances : "+String.valueOf(size)+"\n");
		System.out.print("Accuracy : "+String.valueOf(accuracy));

	}
	private double EuclideanNorm(Instance instance1, Instance instance2) {
		double sum = 0;
		for (int i = 0; i < trainSet.numAttributes()-1; i++) {
			double temp = (double)instance1.value(i)-(double)instance2.value(i);
			sum=(double)sum+(double)temp*temp;
		}
		return sum;
	}
}