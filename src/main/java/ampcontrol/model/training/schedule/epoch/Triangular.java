package ampcontrol.model.training.schedule.epoch;

import ampcontrol.model.visualize.Plot;
import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Triangular learning rate inspired by https://arxiv.org/abs/1506.01186
 *
 * Main difference from dl4j built in implementation is that cycles never stop.
 *
 * @author Christian Skärby
 */
@Data
public class Triangular implements ISchedule {

    private final ISchedule sourceSchedule;
    private final ScheduleType type;
    private final int period;
    private final double step;

    public Triangular(int period,
                      double minLr,
                      double maxLr,
                      ScheduleType type) {
        this(period, 2*(maxLr - minLr) / period, type, new Fixed(minLr));
    }

    public Triangular(@JsonProperty("period") int period,
                      @JsonProperty("step") double step,
                      @JsonProperty("type") ScheduleType type,
                      @JsonProperty("sourceSchedule") ISchedule sourceSchedule) {
        this.sourceSchedule = sourceSchedule;
        this.period = period;
        this.type = type;
        this.step = step;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        // Use abs to create triangular pattern. Increment timeIndex by period/2 to start on positive slope
        final int timeIndex = ((type == ScheduleType.ITERATION ? iteration : epoch) + period / 2) % period;
        return sourceSchedule.valueAt(iteration, epoch) + step *  Math.abs(timeIndex - period / 2);
    }

    @Override
    public ISchedule clone() {
        return new Triangular(period, step, type, sourceSchedule);
    }

    public static void main(String[] args) {
        final int period = 100;
        final double minLr = 1e-5;
        final double maxLr = 1e-2;

        final ISchedule sched = new Triangular(period, minLr, maxLr, ScheduleType.EPOCH);

        List<Double> twoPeriods = IntStream.range(0, 2*period)
                .mapToDouble(epoch -> sched.valueAt(0, epoch))
                .boxed()
                .collect(Collectors.toList());
        DoubleSummaryStatistics stats = twoPeriods.stream().mapToDouble(d->d).summaryStatistics();

        Plot.plot(twoPeriods, "learningrate");
    }
}
