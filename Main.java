import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Main {
    private List<String> trainingNames = new ArrayList<>();
    private List<List<Double>> trainingVectors = new ArrayList<>();
    private List<String> testNames = new ArrayList<>();
    private List<List<Double>> testVectors = new ArrayList<>();
    private Map<String, Integer> categories = new HashMap<>();

    private Map<String, Double> priorProbabilities = new HashMap<>();

    private Map<String, List<double[]>> featureParams = new HashMap<>();

    private Map<String, Double> beforeSmoothing = new HashMap<>();
    private Map<String, Double> afterSmoothing = new HashMap<>();

    public void readFile(String filepath) {
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))){
            String line;
            while((line = br.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");

                List<Double> vectors = new ArrayList<>();
                for (int i = 0; i < parts.length -1; i++) {
                    String number = parts[i].replace(',', '.');
                    vectors.add(Double.parseDouble(number));
                }
                if(filepath.contains("training")){
                    trainingVectors.add(vectors);
                    trainingNames.add(parts[parts.length-1]);
                } else {
                    testVectors.add(vectors);
                    testNames.add(parts[parts.length-1]);
                }
            }

//            System.out.println("Read data from: " + filepath);
        } catch (IOException e){
            System.out.println(e.getMessage());
        }
    }

    public void calculatePriorProbabilities(){
        for(String name : trainingNames) {
            categories.put(name, categories.getOrDefault(name, 0) + 1);
        }

        int totalSamples = trainingNames.size();
        for (Map.Entry<String, Integer> entry : categories.entrySet()) {
            String name = entry.getKey();
            int count = entry.getValue();
            double prior = (double) count / totalSamples;
            priorProbabilities.put(name, prior);
        }

//        System.out.println("\nPrior Probabilities:");
//        for (Map.Entry<String, Double> entry : priorProbabilities.entrySet()) {
//            System.out.println(entry.getKey() + ": " + entry.getValue() +
//                    " (Count: " + categories.get(entry.getKey()) + ")");
//        }
    }

    private double calculateMean(List<Double> values) {
        double sum = 0;
        for(double val : values) {
            sum += val;
        }
        return sum / values.size();
    }

    private double calculateStdDev(List<Double> values, double mean) {
        double sumSquaredDiff = 0;
        for(double val : values) {
            sumSquaredDiff += Math.pow(val - mean, 2);
        }
        double variance = sumSquaredDiff / values.size();
        return Math.sqrt(variance);
    }

    public void calculateLikelihoodParameters() {
        for(String className : categories.keySet()) {
            featureParams.put(className, new ArrayList<>());
        }

        int numFeatures = trainingVectors.get(0).size();

        for(String className : categories.keySet()) {
            List<List<Double>> classSamples = new ArrayList<>();
            for(int i = 0; i < trainingNames.size(); i++) {
                if(trainingNames.get(i).equals(className)) {
                    classSamples.add(trainingVectors.get(i));
                }
            }

            for(int feature = 0; feature < numFeatures; feature++) {
                List<Double> featureValues = new ArrayList<>();
                for(List<Double> sample : classSamples) {
                    featureValues.add(sample.get(feature));
                }

                double mean = calculateMean(featureValues);
                double stdDev = calculateStdDev(featureValues, mean);

                if(stdDev == 0) {
                    stdDev = 0.0001;
                }

                featureParams.get(className).add(new double[]{mean, stdDev});

                if(feature == 0 && !featureValues.isEmpty()) {
                    double firstValue = featureValues.get(0);
                    double prob = calculateGaussianProbability(firstValue, mean, stdDev);
                    beforeSmoothing.put(className, prob);
                }
            }
        }

        System.out.println("Likelihood Parameters (Mean, StdDev):");
        System.out.println();
        for(String className : categories.keySet()) {
            System.out.println("Class: " + className);
            List<double[]> params = featureParams.get(className);
            for(int i = 0; i < params.size(); i++) {
                System.out.printf("Feature %d: Mean = %.4f, StdDev = %.4f%n",
                        i+1, params.get(i)[0], params.get(i)[1]);
            }
        }
    }

    private double calculateGaussianProbability(double x, double mean, double stdDev) {
        double exponent = Math.exp(-Math.pow(x - mean, 2) / (2 * Math.pow(stdDev, 2)));
        return (1 / (Math.sqrt(2 * Math.PI) * stdDev)) * exponent;
    }

    public void applySmoothing() {
        boolean smoothingRequired = false;

        System.out.println("\nProbabilities Before Smoothing:");
        for(String className : categories.keySet()) {
            System.out.printf("%s: %.6f%n", className, beforeSmoothing.get(className));
        }

        for(String className : categories.keySet()) {
            List<double[]> params = featureParams.get(className);
            List<Double> sample = trainingVectors.get(0);
            double totalProb = 0.0;
            boolean featureSmoothingApplied = false;

            for (int i = 0; i < params.size(); i++) {
                double[] featureParams = params.get(i);
                double oldStdDev = featureParams[1];

                if (oldStdDev < 0.1) {
                    double newStdDev = oldStdDev * 1.1;
                    featureParams[1] = newStdDev;
                    smoothingRequired = true;
                    featureSmoothingApplied = true;
                }

                double value = sample.get(i);
                double prob = calculateGaussianProbability(value, featureParams[0], featureParams[1]);
                totalProb += prob;
            }

            if (!smoothingRequired || featureSmoothingApplied) {
                double avgProb = totalProb / params.size();
                afterSmoothing.put(className, avgProb);
            }
        }

        if (!smoothingRequired) {
            System.out.println("No smoothing required. Applying smoothing to the first feature only.");
            applyFirstFeatureSmoothing();
        }

        System.out.println("\nProbabilities After Smoothing:");
        for(String className : categories.keySet()) {
            System.out.printf("%s: %.6f%n", className, afterSmoothing.get(className));
        }
    }

    private void applyFirstFeatureSmoothing() {
        for(String className : categories.keySet()) {
            List<double[]> params = featureParams.get(className);
            if (!params.isEmpty()) {
                double[] firstFeatureParams = params.get(0);
                double oldStdDev = firstFeatureParams[1];
                double newStdDev = oldStdDev * 1.1;
                firstFeatureParams[1] = newStdDev;

                List<Double> sample = trainingVectors.get(0);
                double firstValue = sample.get(0);
                double newProb = calculateGaussianProbability(firstValue, firstFeatureParams[0], newStdDev);
                afterSmoothing.put(className, newProb);
            }
        }
    }

    public String classify(List<Double> sample) {
        String bestClass = null;
        double bestProb = Double.NEGATIVE_INFINITY;

        for(String className : categories.keySet()) {
            double logProb = Math.log(priorProbabilities.get(className));

            List<double[]> params = featureParams.get(className);
            for(int i = 0; i < sample.size(); i++) {
                double value = sample.get(i);
                double mean = params.get(i)[0];
                double stdDev = params.get(i)[1];

                double featureProb = calculateGaussianProbability(value, mean, stdDev);
                featureProb = Math.max(featureProb, 1e-10);
                logProb += Math.log(featureProb);
            }

            if(logProb > bestProb) {
                bestProb = logProb;
                bestClass = className;
            }
        }

        return bestClass;
    }

    public void classifyTestData() {
        int correct = 0;

        Map<String, Map<String, Integer>> confusionMatrix = new HashMap<>();
        for(String actual : categories.keySet()) {
            confusionMatrix.put(actual, new HashMap<>());
            for(String predicted : categories.keySet()) {
                confusionMatrix.get(actual).put(predicted, 0);
            }
        }

        for(int i = 0; i < testVectors.size(); i++) {
            List<Double> sample = testVectors.get(i);
            String actualClass = testNames.get(i);
            String predictedClass = classify(sample);

            confusionMatrix.get(actualClass).put(predictedClass,
                    confusionMatrix.get(actualClass).get(predictedClass) + 1);

            if(predictedClass.equals(actualClass)) {
                correct++;
            }
        }

        double accuracy = (double) correct / testVectors.size() * 100;
        System.out.printf("\nAccuracy: %.2f%%\n", accuracy);

        System.out.println("\nConfusion Matrix:");
        System.out.print("Actual\\Predicted\t");
        for(String className : categories.keySet()) {
            System.out.print(className + "\t");
        }
        System.out.println();

        for(String actual : categories.keySet()) {
            System.out.print(actual + "\t\t");
            for(String predicted : categories.keySet()) {
                System.out.print(confusionMatrix.get(actual).get(predicted) + "\t\t");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        Main main = new Main();

        main.readFile("src/iris_training.txt");
        main.readFile("src/iris_test.txt");

        main.calculatePriorProbabilities();
        main.calculateLikelihoodParameters();
        main.applySmoothing();
        main.classifyTestData();

        Scanner scanner = new Scanner(System.in);
        int numFeatures = main.trainingVectors.get(0).size();

        while(true) {
            System.out.println("\nEnter " + numFeatures + " feature values separated by spaces (or 'exit' to quit):");
            String input = scanner.nextLine().trim();

            if(input.equalsIgnoreCase("exit")) {
                break;
            }

            try {
                String[] values = input.split("\\s+");
                if(values.length != numFeatures) {
                    System.out.println("Error: Please enter exactly " + numFeatures + " values.");
                    continue;
                }

                List<Double> sample = new ArrayList<>();
                for(String value : values) {
                    sample.add(Double.parseDouble(value));
                }

                String predictedClass = main.classify(sample);
                System.out.println("Predicted class: " + predictedClass);

            } catch(NumberFormatException e) {
                System.out.println("Error: Please enter valid numbers.");
            }
        }

        scanner.close();
    }
}