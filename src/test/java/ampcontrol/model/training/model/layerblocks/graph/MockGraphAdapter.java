package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;

/**
 * "Empty" {@link GraphBuilderAdapter} for testing
 *
 * @author Christian Skärby
 */
public class MockGraphAdapter implements GraphBuilderAdapter {

    @Override
    public GraphBuilderAdapter addLayer(String layerName, Layer layer, String... layerInputs) {
        return this;
    }

    @Override
    public GraphBuilderAdapter addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
        return this;
    }

    @Override
    public String mergeIfMultiple(String mergeName, String[] inputs) {
        return inputs.length == 1 ? inputs[0] : mergeName;
    }

    @Override
    public LayerBlockConfig.BlockInfo layer(LayerBlockConfig.BlockInfo info, Layer layer) {
        return info;
    }

    @NotNull
    static ComputationGraph createRealComputationGraph(LayerBlockConfig toTest, int prevNrofOutputs, InputType inputType) {
        final String inputName = "input";
        final ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs(inputName)
                .setInputTypes(inputType);
        final LayerBlockConfig.BlockInfo seb = toTest
                .addLayers(graphBuilder,
                        new LayerBlockConfig.SimpleBlockInfo.Builder()
                                .setInputs(new String[]{inputName})
                                .setPrevNrofOutputs(prevNrofOutputs)
                                .setPrevLayerInd(-1).build());
        // Must have output layer. Must have parameters as well.
        final LayerBlockConfig.BlockInfo output = new DummyOutputLayer().addLayers(graphBuilder, seb);
        graphBuilder.setOutputs(output.getInputsNames());
        final ComputationGraph graph = new ComputationGraph(graphBuilder.build());
        graph.init();
        return graph;
    }
}
