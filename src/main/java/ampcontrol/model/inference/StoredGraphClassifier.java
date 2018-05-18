package ampcontrol.model.inference;

import ampcontrol.audio.ClassifierInputProvider;
import ampcontrol.model.training.model.validation.listen.BestEvalScore;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.nio.file.Paths;

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
	private static final String bestSuffix = "_best";

	StoredGraphClassifier(String path, final ClassifierInputProvider inputProvider) throws IOException {
		final String realFileName = getHashedFileNameFromModelName(path);
		this.model = ModelSerializer.restoreComputationGraph(realFileName, false);

		this.inputProvider = inputProvider;
		this.accuracy = new BestEvalScore(realFileName + ".score").get();
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
		return model.outputSingle(inputProvider.getModelInput());
	}
	
	@Override
	public double getAccuracy() {
		return accuracy;
	}

}
