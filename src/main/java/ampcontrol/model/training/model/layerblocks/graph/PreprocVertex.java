package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.data.iterators.preprocs.Cnn2DtoCnn1DInputPreprocessor;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link PreprocessorVertex} to a graph.
 *
 * @author Christian Skärby
 */
public class PreprocVertex implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(PreprocVertex.class);
    
    private InputPreProcessor preProcessor = new Cnn2DtoCnn1DInputPreprocessor();

    @Override
    public String name() {
        return "pp";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new UnsupportedOperationException("Can only do graphs!");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        log.info("Preproc vertex " + info);
        final int layerInd = info.getPrevLayerInd()+1;
        String thisLayer = info.getName(String.valueOf(layerInd));
        graphBuilder.addVertex(thisLayer, new PreprocessorVertex(preProcessor), info.getInputsNames());
        return new SimpleBlockInfo.Builder(info)
                .setPrevLayerInd(layerInd)
                .setInputs(new String[] {thisLayer})
                .build();
    }


    /**
     * Set the {@link InputPreProcessor} to use.
     * @param preProcessor the {@link InputPreProcessor} to use
     * @return the {@link PreprocVertex}
     */
    public PreprocVertex setPreProcessor(InputPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
        return this;
    }

}
