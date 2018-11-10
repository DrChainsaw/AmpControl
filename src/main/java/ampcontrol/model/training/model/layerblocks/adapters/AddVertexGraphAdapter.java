package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.Predicate;

/**
 * Adds a {@link GraphVertex} right after certain layers.
 *
 * @author Christian Skärby
 */
public class AddVertexGraphAdapter implements GraphBuilderAdapter {

    private static final Logger log = LoggerFactory.getLogger(AddVertexGraphAdapter.class);

    private final GraphBuilderAdapter adapter;
    private final GraphVertex vertexToAdd;
    private final Predicate<String> namesToInsertAfter;
    private final String vertexNamePrefix;

    public AddVertexGraphAdapter(GraphVertex vertexToAdd, Predicate<String> namesToInsertAfter, GraphBuilderAdapter adapter){
        this(adapter, vertexToAdd, namesToInsertAfter, "spy_");
    }

    public AddVertexGraphAdapter(GraphBuilderAdapter adapter, GraphVertex vertexToAdd, Predicate<String> namesToInsertAfter, String vertexNamePrefix) {
        this.adapter = adapter;
        this.vertexToAdd = vertexToAdd;
        this.namesToInsertAfter = namesToInsertAfter;
        this.vertexNamePrefix = vertexNamePrefix;
    }

    @Override
    public GraphBuilderAdapter addLayer(String layerName, Layer layer, String... layerInputs) {
        adapter.addLayer(layerName, layer, layerInputs);
        if(namesToInsertAfter.test(layerName)) {
            throw new IllegalArgumentException("Can't intercept method! No way to give correct output names!");
        }
        return this;
    }

    @Override
    public GraphBuilderAdapter addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
        adapter.addVertex(vertexName, vertex, vertexInputs);
        if(namesToInsertAfter.test(vertexName)) {
            throw new IllegalArgumentException("Can't intercept method! No way to give correct output names!");
        }
        return this;
    }

    @Override
    public String mergeIfMultiple(String mergeName, String[] inputs) {
        return adapter.mergeIfMultiple(mergeName, inputs);
    }

    @Override
    public LayerBlockConfig.BlockInfo layer(LayerBlockConfig.BlockInfo info, Layer layer) {
        final LayerBlockConfig.BlockInfo newInfo = adapter.layer(info, layer);

        final String[] inputs = newInfo.getInputsNames();
        final String[] newInputs = new String[inputs.length];
        final LayerBlockConfig.SimpleBlockInfo.Builder builder = new LayerBlockConfig.SimpleBlockInfo.Builder(newInfo);

        for(int i = 0; i < inputs.length; i++) {

            if (namesToInsertAfter.test(inputs[i])) {
                final String spyName = vertexNamePrefix + inputs[i];
                log.info("Add " + spyName + " with class " + vertexToAdd.getClass());
                adapter.addVertex(spyName, vertexToAdd, inputs[i]);
                builder.setPrevLayerInd(newInfo.getPrevLayerInd() + 1);
                newInputs[i] = spyName;
            } else {
                newInputs[i] = inputs[i];
            }
        }
        return new LayerBlockConfig.SimpleBlockInfo.Builder(newInfo)
                .setInputs(newInputs)
                .build();

    }

}
