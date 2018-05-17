package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;

/**
 * Adapter for {@link NeuralNetConfiguration.ListBuilder}.
 *
 * @author Christian Skärby
 */
public class ListAdapter implements BuilderAdapter {

    private final NeuralNetConfiguration.ListBuilder listBuilder;

    public ListAdapter(NeuralNetConfiguration.ListBuilder listBuilder) {
        this.listBuilder = listBuilder;
    }

    @Override
    public LayerBlockConfig.BlockInfo layer(LayerBlockConfig.BlockInfo info, Layer layer) {
        final int layerInd = info.getPrevLayerInd()+1;
        listBuilder.layer(layerInd, layer);
        return new LayerBlockConfig.SimpleBlockInfo.Builder(info)
                .setPrevLayerInd(layerInd)
                .setInputs(new String[] {info.getName(String.valueOf(layerInd))})
                .build();
    }
}
