package ampcontrol.model.training.schedule.epoch;

import ampcontrol.model.training.schedule.Add;
import ampcontrol.model.training.schedule.MaxLim;
import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * SawTooth learning rate inspired by https://arxiv.org/abs/1506.01186
 * <p>
 * Main difference from dl4j built in implementation is that cycles never stop. Implementation is basically an alias
 * for a bunch of components which together create the triangular pattern since the components by themselves does not
 * explain well what the behaviour is.
 *
 * @author Christian Skärby
 */
@Data
public class SawTooth implements ISchedule {

    private final ISchedule schedule;

    public SawTooth(int period,
                    double min,
                    double max) {
        this(new Cyclic(period, createTriangle(period, min, max)));
    }

    /**
     * Convenience method to create a "triangular" {@link ISchedule} which looks like this:
     * <pre>
     *     max        ^
     *               / \
     *              /   \
     *             /     \
     *     minr   /       \
     *            <------->\
     *              period  \
     *                       \
     * </pre>
     * <p>
     * Typically combined with {@link Cyclic} using "period" to create the cyclic learning rate pattern from the paper
     * cited in class javadoc.
     *
     * @param period See figure above
     * @param min  See figure above
     * @param max  See figure above
     * @return ISchedule with a triangular shape
     */
    public static ISchedule createTriangle(int period, double min, double max) {
        if (min > max) {
            throw new IllegalArgumentException("Min larger than max!!");
        }
        // Crude addition of truncated slopes instead of using abs in order to use simpler components
        // Slope of the lines which make up the "triangle"
        final double slope = 2 * (max - min) / period; // Goes from min to max in period / 2
        final ISchedule posSlope = new Add(new Linear(slope), new Fixed(min)); // y = k*x+m in clunky object notation
        final ISchedule negSlope = new Add(new Linear(-slope), new Fixed(max - min)); // ^^

        return new Add( // Create triangle by adding truncated slopes
                new MaxLim(max, posSlope),  // Will stop increasing after period / 2 since y = max when x = period / 2
                new MaxLim(0, negSlope) // Will start decreasing after period / 2 since y = 0 when x = period / 2
        );
    }

    private SawTooth(
            @JsonProperty("schedule") ISchedule schedule) {
        this.schedule = schedule;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        return schedule.valueAt(iteration, epoch);
    }

    @Override
    public ISchedule clone() {
        return new SawTooth(schedule.clone());
    }
}
