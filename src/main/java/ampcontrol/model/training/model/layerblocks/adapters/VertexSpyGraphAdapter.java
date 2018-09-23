package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.function.Predicate;

/**
 * "Spies" on layers by adding a {@link GraphVertex} right after them.
 *
 * @author Christian Skärby
 */
public class VertexSpyGraphAdapter implements GraphBuilderAdapter {

    private final GraphBuilderAdapter adapter;
    private final GraphVertex vertexSpy;
    private final Predicate<String> namesToSpyOn;
    private final String spyPrefix;

    public VertexSpyGraphAdapter(GraphBuilderAdapter adapter, GraphVertex vertexSpy, Predicate<String> namesToSpyOn){
        this(adapter, vertexSpy, namesToSpyOn, "spy_");
    }

    public VertexSpyGraphAdapter(GraphBuilderAdapter adapter, GraphVertex vertexSpy, Predicate<String> namesToSpyOn, String spyPrefix) {
        this.adapter = adapter;
        this.vertexSpy = vertexSpy;
        this.namesToSpyOn = namesToSpyOn;
        this.spyPrefix = spyPrefix;
    }

    @Override
    public GraphBuilderAdapter addLayer(String layerName, Layer layer, String... layerInputs) {
        adapter.addLayer(layerName, layer, layerInputs);
        if(namesToSpyOn.test(layerName)) {
            throw new IllegalArgumentException("Can't intercept method! No way to give correct output names!");
        }
        return this;
    }

    @Override
    public GraphBuilderAdapter addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
        adapter.addVertex(vertexName, vertex, vertexInputs);
        if(namesToSpyOn.test(vertexName)) {
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
        if(namesToSpyOn.test(newInfo.getInputsNames()[0])) {
            final String spyName = spyPrefix + newInfo.getInputsNames()[0];
            adapter.addVertex(spyName, vertexSpy, newInfo.getInputsNames());
            return new LayerBlockConfig.SimpleBlockInfo.Builder(newInfo)
                    .setPrevLayerInd(newInfo.getPrevLayerInd()+1)
                    .setInputs(new String[] {spyName})
                    .build();
        }
        return newInfo;

    }

}
