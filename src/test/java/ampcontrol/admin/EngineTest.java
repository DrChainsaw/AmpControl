package ampcontrol.admin;

import ampcontrol.admin.service.MockControlRegistry;
import ampcontrol.admin.service.Service;
import ampcontrol.admin.service.control.AppControlService;
import ampcontrol.admin.service.control.ControlRegistry;
import ampcontrol.admin.service.control.SubscriptionRegistry;
import com.beust.jcommander.JCommander;
import org.junit.Test;

import java.util.Collections;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test cases for {@link Engine}
 */
public class EngineTest {

    private final static long sleepTimeMs = 10;
    private final static int maxNrofRetries = 100;

    private final static String exitCmdPar = "-exitCmd";
    private final static String exitCmdMsg = "frehththtyjy";


    /**
     * Test that engine can be started and stopped
     */
    @Test
    public void startStopEngine() {
        final Engine engine = new Engine();
        final MockAppControl appCtrl = new MockAppControl();
        engine.initialize(appCtrl, Collections.singleton(new MockService()));
        appCtrl.assertRunningState(false);

        new Thread(engine::run).start();

        try {
            Thread.sleep(sleepTimeMs);

            appCtrl.assertRunningState(true);

            engine.stop();

            Thread.sleep(sleepTimeMs);

            appCtrl.assertRunningState(false);

        } catch (InterruptedException e) {
            fail("test interrupted!");
        }
    }

    /**
     * Test that service can be started and stopped (and restarted)
     */
    @Test
    public void startStopService() {
        final Engine engine = new Engine();
        String argStr = exitCmdPar + " " + exitCmdMsg;
        JCommander.newBuilder().addObject(engine)
                .build()
                .parse(argStr.split(" "));

        final MockService service = new MockService();
        final MockAppControl appControl = new MockAppControl();
        engine.initialize(appControl, Collections.singleton(service));
        service.assertRunningState(false);

        new Thread(engine::run).start();

        waitForCondition(() -> appControl.getM2aMap().isRegistered(service.start));

        service.assertRunningState(false);

        appControl.getM2aMap().execute(service.start);

        waitForCondition(() -> service.isStarted);

        service.assertRunningState(true);
        appControl.getM2aMap().execute(service.stop);

        waitForCondition(() -> !service.isStarted);

        service.assertRunningState(false);
        appControl.getM2aMap().execute(service.start);

        waitForCondition(() -> service.isStarted);

        service.assertRunningState(true);

        appControl.getM2aMap().execute(exitCmdMsg);

        waitForCondition(() -> !service.isStarted);
        service.assertRunningState(false);
    }

    private static void waitForCondition(Supplier<Boolean> condition) {
        try {
            int retryCnt = 0;
            while (!condition.get() && retryCnt < maxNrofRetries) {
                Thread.sleep(sleepTimeMs);
                retryCnt++;
            }
        } catch (InterruptedException e) {
            fail("test interrupted!");
        }
    }

    private final static class MockAppControl implements AppControlService {

        private final MockControlRegistry m2aMap;
        private boolean isStarted = false;

        MockAppControl() {
            m2aMap = new MockControlRegistry();
        }

        @Override
        public ControlRegistry start() {
            isStarted = true;
            return m2aMap;
        }

        @Override
        public void stop() {
            isStarted = false;
        }

        MockControlRegistry getM2aMap() {
            return m2aMap;
        }

        void assertRunningState(boolean expected) {
            assertEquals("Incorrect running state!", expected, isStarted);
        }
    }


    private final static class MockService implements Service {

        private final String start;
        private final String stop;

        private boolean isStarted = false;

        MockService() {
            this("start", "stop");
        }

        MockService(String start, String stop) {
            this.start = start;
            this.stop = stop;
        }

        @Override
        public void stop() {
            isStarted = false;
        }

        @Override
        public void registerTo(SubscriptionRegistry subscriptionRegistry) {
            subscriptionRegistry.registerSubscription(start, () -> isStarted = true);
            subscriptionRegistry.registerSubscription(stop, () -> isStarted = false);
        }

        void assertRunningState(boolean expected) {
            assertEquals("Incorrect running state!", expected, isStarted);
        }
    }
}