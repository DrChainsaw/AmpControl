package ampcontrol.model.training.model.evolve.fitness;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import org.apache.commons.lang.mutable.MutableDouble;
import org.apache.commons.lang3.mutable.Mutable;
import org.apache.commons.lang3.mutable.MutableObject;
import org.junit.Test;
import org.slf4j.LoggerFactory;

import java.util.function.Consumer;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link LogFitness}
 *
 * @author Christian Skärby
 */
public class LogFitnessTest {

    /**
     * Test that fitness is logged and that candidate is returned
     */
    @Test
    public void apply() {
        Mutable<String> logMessage = new MutableObject<>();
        final AppenderBase<ILoggingEvent> appender = new AppenderBase<ILoggingEvent>() {
            @Override
            protected void append(ILoggingEvent iLoggingEvent) {
                logMessage.setValue(iLoggingEvent.getLoggerName() +": " + iLoggingEvent.getMessage());
            }
        };
        appender.start();

        final ch.qos.logback.classic.Logger logger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(MockPolicy.class);
        logger.setAdditive(false);
        logger.addAppender(appender);
        final Level prevLevel = logger.getLevel();
        logger.setLevel(Level.INFO);

        final FitnessPolicy<Double> policy =  new LogFitness<>(new MockPolicy());
        final MutableDouble actualFitness = new MutableDouble(-1);
        final double retCand =policy.apply(666d, actualFitness::setValue);

        logger.detachAppender(appender);
        logger.setLevel(prevLevel);

        assertEquals("Incorrect candidate returned!", 666d, retCand,1e-10);
        assertEquals("Incorrect fitness reported!", 666d, actualFitness.doubleValue(), 1e-10);
        assertEquals("Incorrect message received!",
                MockPolicy.class.getName() + ": " + "got fitness: 666.0",
                logMessage.getValue());
    }

    private final static class MockPolicy implements FitnessPolicy<Double> {
        @Override
        public Double apply(Double candidate, Consumer<Double> fitnessListener) {
            fitnessListener.accept(candidate);
            return candidate;
        }
    }
}