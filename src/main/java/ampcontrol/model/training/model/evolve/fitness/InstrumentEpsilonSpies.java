package ampcontrol.model.training.model.evolve.fitness;

import ampcontrol.model.training.listen.ActivationContribution;
import ampcontrol.model.training.model.CompGraphAdapter;
import ampcontrol.model.training.model.evolve.mutate.util.Graph;
import ampcontrol.model.training.model.evolve.mutate.util.TraverseBuilder;
import ampcontrol.model.training.model.evolve.selection.ModelComparatorRegistry;
import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import org.apache.commons.lang3.mutable.Mutable;
import org.apache.commons.lang3.mutable.MutableObject;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Comparator;
import java.util.function.Consumer;

/**
 * Adds an {@link ActivationContribution} for every layer which is followed by an {@link EpsilonSpyVertex} and registers
 * an {@link ActivationContributionComparator} to the provided comparatorRegistry.
 *
 * @param <T>
 * @author Christian Skärby
 */
public class InstrumentEpsilonSpies<T extends CompGraphAdapter> implements FitnessPolicy<T> {

    private static final Logger log = LoggerFactory.getLogger(InstrumentEpsilonSpies.class);

    private final ModelComparatorRegistry comparatorRegistry;

    private final static class ActivationContributionComparator implements Consumer<INDArray>, Comparator<Integer> {

        private double[] activationContribution = null;


        @Override
        public int compare(Integer elem1, Integer elem2) {
            return -Double.compare(
                    activationContribution[elem1],
                    activationContribution[elem2]);
        }

        @Override
        public void accept(INDArray activationContribution) {
            //log.debug("Got contrib: " + activationContribution);
            final double[] contrib = activationContribution.toDoubleVector();
                if (this.activationContribution == null) {
                    this.activationContribution = contrib;
                } else {
                    for(int i = 0; i < contrib.length; i++) {
                        this.activationContribution[i] += contrib[i];
                }
            }
        }
    }

    public InstrumentEpsilonSpies(ModelComparatorRegistry comparatorRegistry) {
        this.comparatorRegistry = comparatorRegistry;
    }

    @Override
    public T apply(T candidate, Consumer<Double> fitnessListener) {
        final Mutable<String> parentVertex = new MutableObject<>();
        final ComputationGraphConfiguration config = candidate.asModel().getConfiguration();

        // Get rid of ModelComparatorRegistry and instead add ActivationContributionComparator directly here?
        // Would most likely require that type is EvolvingGraphAdapter

        final Graph<String> graph = TraverseBuilder.forwards(config)
                .enterListener(parentVertex::setValue)
                .visitListener(vertex -> {
                    if (config.getVertices().get(vertex) instanceof EpsilonSpyVertex) {
                        final String spiedVertex = parentVertex.getValue();
                        createComparator(candidate, spiedVertex);
                    }
                })
                .build();

        candidate.asModel().getConfiguration().getNetworkInputs().stream()
                .flatMap(graph::children).forEach(vertex -> {/* Everything happens in listener above */});
        return candidate;
    }

    private void createComparator(T candidate, String spiedVertex) {
        log.debug("Instrument " + spiedVertex + " with activation contribution listener");
        final ActivationContributionComparator comparator = new ActivationContributionComparator();
        candidate.asModel().addListeners(new ActivationContribution(spiedVertex, comparator));

        switch (candidate.asModel().getVertex(spiedVertex).getLayer().type()) {
            case CONVOLUTIONAL:
            case CONVOLUTIONAL3D:
                comparatorRegistry.add(candidate.asModel(), spiedVertex, 0, comparator);
                break;
            case FEED_FORWARD:
                comparatorRegistry.add(candidate.asModel(), spiedVertex, 1, comparator);
                break;
            default:
                throw new UnsupportedOperationException("layertype not implemented: " + candidate.asModel().getVertex(spiedVertex).getLayer().type());
        }
    }
}
