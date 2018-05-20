package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Global mean max. Does (mean(A) + max(A)) / 2 where A is the activation from the input.
 * TODO: Add reference to paper with evaluation. Can't remember where I saw it...
 *
 * @author Christian Skärby
 */
public class GlobMeanMax implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(GlobMeanMax.class);

    @Override
    public String name() {
        return "gpmm";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new UnsupportedOperationException("Must work with a graphbuilder");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {

        log.info("Global mean max " + info);
        int nextLayerInd = info.getPrevLayerInd()+1;

        final String maxPoolName = info.getName("gp_max" + String.valueOf(nextLayerInd++));
        graphBuilder.addLayer(maxPoolName,
                new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.MAX)
                        .build(), info.getInputsNames());

        final String avgPoolName = info.getName("gp_mean" + String.valueOf(nextLayerInd++));
        graphBuilder.addLayer(avgPoolName,
                new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.AVG)
                        .build(), info.getInputsNames());

        final String addVertexName = info.getName("gp_add" + String.valueOf(nextLayerInd++));
        graphBuilder.addVertex(addVertexName,
                new ElementWiseVertex(ElementWiseVertex.Op.Add), maxPoolName, avgPoolName);

        final String scaleVertexName = info.getName("gp_sc" + String.valueOf(nextLayerInd++));
        graphBuilder.addVertex(scaleVertexName, new ScaleVertex(0.5), addVertexName);

        return new SimpleBlockInfo.Builder(info)
                .setPrevLayerInd(nextLayerInd)
                .setInputs(new String[] {scaleVertexName}).build();
    }
}
