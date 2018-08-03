package ampcontrol.model.training.schedule.epoch;

import ampcontrol.model.training.schedule.Add;
import ampcontrol.model.training.schedule.MaxLim;
import ampcontrol.model.visualize.Plot;
import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
                    double minLr,
                    double maxLr) {
        this(new Cyclic(period, createTriangle(period, minLr, maxLr)));
    }

    /**
     * Convenience method to create a "triangular" {@link ISchedule} which looks like this:
     * <pre>
     *     maxLr      ^
     *               / \
     *              /   \
     *             /     \
     *     minrLr /       \
     *            <------->\
     *              period  \
     *                       \
     * </pre>
     *
     * Typically combined with {@link Cyclic} using "period" to create the cyclic learning rate pattern from the paper
     * cited in class javadoc.
     *
     * @param period See figure above
     * @param minLr See figure above
     * @param maxLr See figure above
     * @return ISchedule with a triangular shape
     */
    public static ISchedule createTriangle(int period, double minLr, double maxLr) {
        // Slope of the lines which make up the "triangle"
        final double slope = 2*(maxLr - minLr) / period; // Goes from minLr to maxLr in period / 2
        final ISchedule posSlope = new Add(new Linear(slope), new Fixed(minLr)); // y = k*x+m in clunky object notation
        final ISchedule negSlope = new Add(new Linear(-1 * slope), new Fixed(maxLr - minLr)); // ^^

        return new Add( // Create triangle by adding truncated slopes
                new MaxLim(maxLr, posSlope),  // Will stop increasing after period / 2 since y = maxLr when x = period / 2
                new MaxLim(0, negSlope) // Will start decreasing after period / 2 since y = 0 when x = period / 2
        );
    }

    private static double calcCoeff(int period,
                                    double minLr,
                                    double maxLr) {
        return 2*(maxLr - minLr) / period;
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

    public static void main(String[] args) {
        final int period = 100;
        final double minLr = -1;
        final double maxLr = 10;

        final ISchedule sched = new SawTooth(period, minLr, maxLr);

        List<Double> twoPeriods = IntStream.range(0, 2 * period)
                .mapToDouble(epoch -> sched.valueAt(0, epoch))
                .boxed()
                .collect(Collectors.toList());
        DoubleSummaryStatistics stats = twoPeriods.stream().mapToDouble(d -> d).summaryStatistics();
        Plot.plot(twoPeriods, "learningrate");
    }
}
