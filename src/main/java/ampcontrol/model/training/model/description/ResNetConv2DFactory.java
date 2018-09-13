package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.model.GenericModelHandle;
import ampcontrol.model.training.model.GraphModelAdapter;
import ampcontrol.model.training.model.ModelHandle;
import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.builder.DeserializingModelBuilder;
import ampcontrol.model.training.model.builder.ModelBuilder;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.Scale;
import ampcontrol.model.training.model.layerblocks.graph.SeBlock;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import ampcontrol.model.training.schedule.MinLim;
import ampcontrol.model.training.schedule.Mul;
import ampcontrol.model.training.schedule.epoch.Exponential;
import ampcontrol.model.training.schedule.epoch.Offset;
import ampcontrol.model.training.schedule.epoch.SawTooth;
import ampcontrol.model.training.schedule.epoch.Step;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Description of some homebrewed resnets with 2D convolutions
 *
 * @author Christian Skärby
 */
public class ResNetConv2DFactory {
    private final MiniEpochDataSetIterator trainIter;
    private final MiniEpochDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final FileNamePolicy modelFileNamePolicy;

    public ResNetConv2DFactory(MiniEpochDataSetIterator trainIter, MiniEpochDataSetIterator evalIter, int[] inputShape, String namePrefix, FileNamePolicy modelFileNamePolicy) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
        this.inputShape = inputShape;
        this.namePrefix = namePrefix;
        this.modelFileNamePolicy = modelFileNamePolicy;
    }

    /**
     * Adds the ModelHandles defined by this class to the given list
     *
     * @param modelData list to add models to
     */
    public void addModelData(List<ModelHandle> modelData) {

        final LayerBlockConfig pool = new Pool2D().setSize(3).setStride(3);
        final int resblockOutFac = 1;
        // final LayerBlockConfig pool = new MinMaxPool().setSize(3).setStride(3); final int resblockOutFac = 2;

        final int schedPeriod = 50;
        final ISchedule lrSched = new Mul(new MinLim(0.02, new Step(4000, new Exponential(0.2))),
                new SawTooth(schedPeriod, 1e-6, 0.1));
        final ISchedule momSched = new Offset(schedPeriod / 2,
                new SawTooth(schedPeriod, 0.85, 0.95));

        IntStream.of(5, 10).forEach(resDepth ->
                Stream.of(new IdBlock(), new SeBlock()).forEach(seOrId ->
                        DoubleStream.of(0.002).forEach(lambda -> {
                            ModelBuilder builder = new DeserializingModelBuilder(modelFileNamePolicy,
                                    new BlockBuilder()
                                            .setNamePrefix(namePrefix)
                                            .setUpdater(new Nesterovs(lrSched, momSched))
                                            .first(new ConvType(inputShape))

                                            .andThen(new Conv2DBatchNormBetween()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize(3)
                                                    .setNrofKernels(64))
                                            .andThen(pool)
                                            .andThen(new Conv2DBatchNormBetween()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize(3)
                                                    .setNrofKernels(128))
                                            .andThen(pool)
                                           // .andThen(seOrId)
                                            .andThen(new Conv2DBatchNormBetween()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize(3)
                                                    .setNrofKernels(128))
                                            .andThen(pool)
                                           // .andThen(seOrId)
                                            .andThenStack(resDepth)
                                            .aggRes()
                                            .aggOf(new Conv2DBatchNormBetween()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(64))
                                            .andThen(new Conv2DBatchNormBetween()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize(3)
                                                    .setNrofKernels(128))
                                            .andThen(new Conv2DBatchNormBetween()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(128 * resblockOutFac))
                                            .andFinally(new Scale(0.1))
                                            .andFinally(seOrId)
                                            //.andFinally(new DropOut().setDropProb(dropOutProb))
                                            .andThenStack(2)
                                            .of(new Dense())
                                            .andFinally(new CenterLossOutput(trainIter.totalOutcomes())
                                                    .setAlpha(0.6)
                                                    .setLambda(lambda)));
                            modelData.add(new GenericModelHandle(
                                    trainIter,
                                    evalIter,
                                    new GraphModelAdapter(builder.buildGraph()),
                                    builder.name()));
                        })
                )
        );
    }
}
