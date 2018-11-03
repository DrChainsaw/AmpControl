package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;

import java.util.function.Predicate;

/**
 * Various hideous utility functions around {@link ComputationGraphConfiguration.GraphBuilder} to assist in traversal.
 */
public class GraphBuilderUtil {

    /**
     * Returns true if a change of size (NOut or NIn) propagates to the (next or prev) layers (NIn or NOut).
     * For example: If NOut is changed for conv1 in case conv1 -> maxpool -> conv2, then it is conv2 which
     * needs to change its NIn as the maxpool is transparent w.r.t sizes. The converse goes for the case
     * when NIn is changed in conv2; then conv1 needs to change NOut for the very same reason.
     *
     * @param builder Configuration to check against
     * @return Predicate which is true if size change propagates
     */
    public static Predicate<String> changeSizePropagates(ComputationGraphConfiguration.GraphBuilder builder) {
        return vertex -> doesSizeChangePropagate(builder.getVertices().get(vertex));
    }

    private static boolean doesSizeChangePropagate(GraphVertex vertex) {
        if (!(vertex instanceof LayerVertex)) {
            return true;
        }
        LayerVertex layerVertex = (LayerVertex) vertex;

        if (layerVertex.numParams(false) == 0) {
            return true;
        }


        return layerVertex.getLayerConf().getLayer() instanceof BatchNormalization;
    }
}
