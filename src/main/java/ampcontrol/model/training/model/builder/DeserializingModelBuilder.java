package ampcontrol.model.training.model.builder;

import ampcontrol.model.training.model.naming.FileNamePolicy;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * {@link ModelBuilder} which loads the model from a file if a model with the same name as a source {@link ModelBuilder}
 * is found in the given directory. If not found, the source {@link ModelBuilder} is asked to build the model.
 *
 * @author Christian Skärby
 */
public class DeserializingModelBuilder implements ModelBuilder {

    private static final Logger log = LoggerFactory.getLogger(DeserializingModelBuilder.class);

    private final ModelBuilder sourceBuilder;
    private final File modelFile;

    /**
     * Constructor
     *
     * @param modelFileNamePolicy how to determine the filename from a model name
     * @param sourceBuilder {@link ModelBuilder} for which this class will try to restore a serialized model.
     */
    public DeserializingModelBuilder(FileNamePolicy modelFileNamePolicy, ModelBuilder sourceBuilder) {
        this.sourceBuilder = sourceBuilder;
        modelFile = new File(modelFileNamePolicy.toFileName(name()));
        log.info("filename: " + modelFile.getAbsolutePath());
    }

    @Override
    public MultiLayerNetwork build() {
        if (modelFile.exists()) {
            try {
                log.info("restoring saved model: " + modelFile.getAbsolutePath());
                // load updater with Adam results in score NaN for some unknown reason
                return ModelSerializer.restoreMultiLayerNetwork(modelFile, !name().contains("Adam"));
            } catch (IOException e) {
                throw new IllegalStateException("Failed to load model");
            }
        }
        return sourceBuilder.build();
    }

    @Override
    public ComputationGraph buildGraph() {
        if (modelFile.exists()) {
            try {
                log.info("restoring saved model: " + modelFile.getAbsolutePath());
                // load updater with Adam results in score NaN for some unknown reason
                return ModelSerializer.restoreComputationGraph(modelFile, !name().contains("Adam"));
            } catch (IOException e) {
                throw new IllegalStateException("Failed to load model " + modelFile, e);
            }
        }
        return sourceBuilder.buildGraph();
    }

    @Override
    public String name() {
        return sourceBuilder.name();
    }

}
