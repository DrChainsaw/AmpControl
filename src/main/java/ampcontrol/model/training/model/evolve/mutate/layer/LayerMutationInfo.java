package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.mutate.util.GraphBuilderUtil;
import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

import java.util.function.BiFunction;

/**
 * Base class for mutating vertices. Apart from name itself, it contains functions to calculate nIn and nOut
 * for a given layer in a given graph.
 *
 * @author Christian Skärby
 */
@Builder
@Getter
public class LayerMutationInfo {

    private final String layerName;

    private final BiFunction<String, ComputationGraphConfiguration.GraphBuilder, Long> outputSizeMapping =
            GraphBuilderUtil::getOutputSize;

    private final BiFunction<String, ComputationGraphConfiguration.GraphBuilder, Long> inputSizeMapping =
            GraphBuilderUtil::getInputSize;

}
