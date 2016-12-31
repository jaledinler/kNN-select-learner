import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Attribute;
import weka.core.Instances;

public class Knn_select {
    static int numOfAttributes;
	public static void main(String[] args) throws IOException {
		if(args.length != 5) {
			System.out.println(args.length);
			System.err.println("Usage: Java DecisionTree"
					+ "<train-set-file> <test-set-file> threshold");
			System.exit(1);;
		}
		BufferedReader train = new BufferedReader(
				new FileReader(args[0]));
		BufferedReader test = new BufferedReader(
				new FileReader(args[1]));
		Instances trainSet = new Instances(train);
		Instances testSet = new Instances(test);
		Knn_selectImpl knn = new Knn_selectImpl(trainSet,testSet,Integer.parseInt(args[2]), Integer.parseInt(args[3]), Integer.parseInt(args[4]));
		trainSet.setClassIndex(trainSet.numAttributes() - 1);
		Attribute att = trainSet.attribute(trainSet.numAttributes()-1);
		String name = att.name();
        if (name.equals("class"))
        	knn.printCl();
        if (name.equals("response"))
        	knn.printReg();
		train.close();
		test.close();

	}

}
