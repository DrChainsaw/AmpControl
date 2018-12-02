package ampcontrol.model.training.model.evolve.fitness;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.Consumer;

/**
 * Logs fitness values
 * @param <T>
 *
 * @author Christian Skärby
 */
public class LogFitness<T> implements FitnessPolicy<T> {

    private final Logger log;
    private final FitnessPolicy<T> source;

    public LogFitness(FitnessPolicy<T> source) {
        this(LoggerFactory.getLogger(source.getClass()), source);
    }

    public LogFitness(Logger log, FitnessPolicy<T> source) {
        this.log = log;
        this.source = source;
    }

    @Override
    public T apply(T candidate, Consumer<Double> fitnessListener) {
        return source.apply(candidate, fitness -> {
            log.info("got fitness: " + fitness);
            fitnessListener.accept(fitness);
        });
    }
}
