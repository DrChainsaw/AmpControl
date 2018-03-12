package ampControl.admin;

import com.beust.jcommander.JCommander;
import ampControl.admin.service.control.AppControlService;
import ampControl.admin.service.control.MessageToActionMap;
import ampControl.admin.service.Service;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Test cases for {@link Engine}
 */
public class EngineTest {

    private final static long sleepTimeMs = 5;

    private final static String autoActPar = "-mqttActAutoMsg";
    private final static String autoActMsg = "frehththtyjy";

    private final static String autoDeactPar = "-mqttDectAutoMsg";
    private final static String autoDeactMsg = "fgrhjgkds";

    /**
     * Test that engine can be started and stopped
     */
    @Test
    public void startStopEngine() {
        final Engine engine = new Engine();
        final MockAppControl appCtrl = new MockAppControl();
        engine.initialize(appCtrl, new MockService());
        appCtrl.assertRunningState(false);

        new Thread(() -> engine.run()).start();

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
        String argStr = autoActPar + " " + autoActMsg + " " + autoDeactPar + " " + autoDeactMsg;
        JCommander.newBuilder().addObject(engine)
                .build()
                .parse(argStr.split(" "));

        final MockService service = new MockService();
        final MockAppControl appControl = new MockAppControl();
        engine.initialize(appControl, service);
        service.assertRunningState(false);

        new Thread(() -> engine.run()).start();

        try {
            Thread.sleep(sleepTimeMs);

            service.assertRunningState(false);

            appControl.getM2aMap().runAction(autoActMsg);

            Thread.sleep(sleepTimeMs);

            service.assertRunningState(true);
            appControl.getM2aMap().runAction(autoDeactMsg);

            Thread.sleep(sleepTimeMs);

            service.assertRunningState(false);
            appControl.getM2aMap().runAction(autoActMsg);

            Thread.sleep(sleepTimeMs);

            service.assertRunningState(true);

            engine.stop();

            Thread.sleep(sleepTimeMs);
            service.assertRunningState(false);

        } catch (InterruptedException e) {
            fail("test interrupted!");
        }
    }

    private final static class MockAppControl implements AppControlService {

        private final MockM2aMap m2aMap;
        private boolean isStarted = false;

        MockAppControl() {
            m2aMap = new MockM2aMap();
        }

        @Override
        public MessageToActionMap start() {
            isStarted = true;
            return m2aMap;
        }

        @Override
        public void stop() {
            isStarted = false;
        }

        MockM2aMap getM2aMap() {
            return m2aMap;
        }

        void assertRunningState(boolean expected) {
            assertEquals("Incorrect running state!", expected, isStarted);
        }
    }

    private final static class MockM2aMap implements MessageToActionMap {

        private final Map<String, Runnable> actions = new HashMap<>();

        @Override
        public void mapMessage(String message, Runnable action) {
            actions.put(message, action);
        }

        @Override
        public void setConnectionFailedAction(Runnable action) {
            // Ignore
        }

        void runAction(String actionMessage) {
            actions.get(actionMessage).run();
        }
    }

    private final static class MockService implements Service {

        private boolean isStarted = false;

        @Override
        public void start() {
            isStarted = true;
        }

        @Override
        public void stop() {
            isStarted = false;
        }

        void assertRunningState(boolean expected) {
            assertEquals("Incorrect running state!", expected, isStarted);
        }
    }
}