package ampControl.model.inference;

import ampControl.audio.ClassifierInputProvider;
import ampControl.model.training.data.iterators.preprocs.CnnToManyToOneRnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.function.Function;
import java.util.regex.Pattern;

/**
 * {@link Classifier} which loads a stored {@link ComputationGraph}.
 * TODO: Add testcase with some minimal model(s)
 *
 * @author Christian Skärby
 */
public class StoredGraphClassifier implements Classifier {

	private final ComputationGraph model;
	private final ClassifierInputProvider inputProvider;
	private final double accuracy;
	private final Function<INDArray, INDArray> postProcessor;

	private static final String bestSuffix = "_best";

	StoredGraphClassifier(String path, final ClassifierInputProvider inputProvider) throws IOException {
		String realFileName = getHashedFileNameFromModelName(path);
		model = ModelSerializer.restoreComputationGraph(realFileName, false);

		// TODO: Probably obsolete by now
		if(path.contains("LstmTimeSeq")) {
			this.inputProvider = () ->  CnnToManyToOneRnnPreProcessor.cnnToRnnFeature(inputProvider.getModelInput().dup());

			this.postProcessor = output -> {
				final long lastTimeStep = output.size(2)-1;
				return output.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(lastTimeStep));
			};
		} else {		
			this.inputProvider= inputProvider;
			this.postProcessor = Function.identity();
		}
		this.accuracy = getAccuracy(path);
	}

	@NotNull
	private static String getHashedFileNameFromModelName(String path) {
		final String modelName = Paths.get(path).getFileName().toString();
		String pathHashCode = modelName;
		String suffix = "";
		if(pathHashCode.endsWith(bestSuffix)) {
			pathHashCode = pathHashCode.replace(bestSuffix, "");
			suffix = bestSuffix;
		}
		return path.replace(modelName, "") + pathHashCode.hashCode() + suffix;
	}

	@Override
	public INDArray classify() {
		return postProcessor.apply(model.outputSingle(inputProvider.getModelInput()));
	}
	
	@Override
	public double getAccuracy() {
		return accuracy;
	}
		
	
    private final static Pattern accPattern = Pattern.compile("\\s*Accuracy:\\s*(\\d*,?\\d*)");	
	public static double getAccuracy(String path) throws IOException {
        BufferedReader scoreReader = Files.newBufferedReader(Paths.get(getHashedFileNameFromModelName(path) + ".score"));

        return scoreReader.lines()
                .map(line -> accPattern.matcher(line))
                .filter(matcher -> matcher.matches())
                .map(matcher -> matcher.group(1))
                .map(str -> str.replace(",", "."))
                .mapToDouble(str -> Double.parseDouble(str))
                .findFirst()
                .orElse(0);		
	}
}
