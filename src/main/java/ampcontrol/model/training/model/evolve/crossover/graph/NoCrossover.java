package ampcontrol.model.training.model.evolve.crossover.graph;

import ampcontrol.model.training.model.evolve.crossover.Crossover;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Does not do any crossover
 *
 * @author Christian Skärby
 */
public class NoCrossover implements Crossover<GraphInfo> {

    private static final Logger log = LoggerFactory.getLogger(NoCrossover.class);

    @Override
    public GraphInfo cross(GraphInfo first, GraphInfo second) {
        log.info("No crossover performed");
        return new GraphInfo.NoopResult(first);
    }
}
