import React, {useEffect, useEffectEvent, useState} from 'react';
import {
  Alert, AppState, BackHandler, Linking, NativeEventEmitter, NativeModules,
  PermissionsAndroid, Pressable, requireNativeComponent, ScrollView, StatusBar,
  StyleSheet, Text, useWindowDimensions, View,
} from 'react-native';
import type {ReactNode} from 'react';
import type {ViewProps} from 'react-native';
import SetupArt from './assets/camera-setup.svg';
import LiveArt from './assets/live-recognition.svg';

const colors = {
  bg: '#f5f7f6', text: '#172c29', green: '#116b59', muted: '#5d706a',
  pale: '#e0eee7', border: '#dfe6e1', camera: '#223a33', white: '#ffffff',
};

type DurationRow = {id: string; name: string; duration: number};
type Summary = {id: string; started: number; duration: number; unclassified: number; exercises: DurationRow[]};
type State = {
  phase: 'idle' | 'setup' | 'countdown' | 'active' | 'paused' | 'summary' | 'saved';
  ready: boolean; elapsed: number; countdown: number; label: string | null;
  message: string | null; error: string | null; canFinish: boolean; draft: Summary | null;
};
type Route = 'home' | 'guide' | 'flow' | 'history' | 'detail';
type Permission = 'unknown' | 'granted' | 'denied' | 'blocked';

const native = NativeModules.FormLiteRecognition as {
  addListener: (event: string) => void; removeListeners: (count: number) => void;
  insets: () => Promise<string>; state: () => Promise<string>; openSetup: () => void; closeSetup: () => void;
  switchCamera: () => void; start: () => void; pause: () => void; resume: () => void;
  finish: () => void; discard: () => void; save: () => Promise<string>;
  sessions: () => Promise<string>; details: (id: string) => Promise<string>;
  deleteSession: (id: string | null) => Promise<void>;
};
const NativeCamera = requireNativeComponent<ViewProps>('FormLiteCamera');
const cameraPermission = PermissionsAndroid.PERMISSIONS.CAMERA;
const initial: State = {
  phase: 'idle', ready: false, elapsed: 0, countdown: 0, label: null,
  message: null, error: null, canFinish: false, draft: null,
};

const guide = [
  ['Side plank', 'Set the camera low so your shoulders, hips, and feet stay in frame.'],
  ['Squats', 'Step back until the camera can see you from shoulders to feet.'],
  ['Reverse crunches', 'Place the camera low and keep your torso and legs visible.'],
  ['Air jump rope', 'Leave room above your head and keep your hands and feet visible.'],
];

function time(ms: number) {
  const seconds = Math.floor(ms / 1000);
  const hours = Math.floor(seconds / 3600);
  const parts = [Math.floor(seconds / 60) % 60, seconds % 60].map(part => String(part).padStart(2, '0'));
  return hours ? [hours, ...parts].join(':') : parts.join(':');
}

function Button({title, press, outline = false, disabled = false}: {
  title: string; press: () => void; outline?: boolean; disabled?: boolean;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{disabled}}
      disabled={disabled}
      onPress={() => press()}
      style={({pressed}) => [styles.button, outline && styles.outline, (disabled || pressed) && styles.dim]}>
      <Text style={[styles.buttonText, outline && styles.outlineText]}>{title}</Text>
    </Pressable>
  );
}

function Header({title, back, trailing, compact = false}: {title: string; back?: () => void; trailing?: ReactNode; compact?: boolean}) {
  return (
    <View style={[styles.header, compact && styles.liveHeader]}>
      {back ? <Pressable accessibilityRole="button" accessibilityLabel="Back" onPress={back} style={styles.headingPress}>
        <Text style={styles.heading}>‹  {title}</Text>
      </Pressable> : <Text accessibilityRole="header" style={styles.heading}>{title}</Text>}
      {trailing}
    </View>
  );
}

function Tabs({active, home, history}: {active: 'home' | 'history'; home: () => void; history: () => void}) {
  return <View style={[styles.row, styles.tabs]}>
    <Button title="Home" press={home} outline={active !== 'home'} />
    <Button title="History" press={history} outline={active !== 'history'} />
  </View>;
}

function TimerCard({duration}: {duration: number}) {
  return <View style={styles.tint}>
    <Text style={styles.tag}>ACTIVE SESSION TIME</Text>
    <Text style={styles.largeTime}>{time(duration)}</Text>
  </View>;
}

function CameraPane({active, ready, enabled, countdown}: {
  active: boolean; ready: boolean; enabled: boolean; countdown: number;
}) {
  const {width} = useWindowDimensions();
  const scale = Math.min(1, (width - 48) / 342);
  const art = active ? <LiveArt /> : <SetupArt />;
  return <View style={[styles.camera, {height: 340 * scale + (active ? 41 : 33)}]}>
    {enabled && <NativeCamera style={styles.preview} />}
    {!ready && <View pointerEvents="none" style={[StyleSheet.absoluteFill, styles.cameraCover]}>
      <View style={[styles.illustration, {top: 4 + (340 * scale - 340) / 2, transform: [{scale}]}]}>{art}</View>
      <Text style={styles.previewCaption}>Illustrative camera preview</Text>
    </View>}
    {countdown > 0 && <View pointerEvents="none" style={styles.countdown}>
      <Text accessibilityLabel={`Starting in ${countdown}`} style={styles.countdownText}>{countdown}</Text>
    </View>}
  </View>;
}

function DurationTable({summary}: {summary: Summary}) {
  const rows = [...summary.exercises, {id: 'unclassified', name: 'Unclassified time', duration: summary.unclassified}];
  return <View style={styles.durationTable}>
    {summary.exercises.length === 0 && <Text style={styles.secondary}>No supported exercise was confidently recognized.</Text>}
    {rows.map(row => <View key={row.id} style={[styles.row, styles.durationRow]}>
      <Text style={styles.durationLabel}>{row.name}</Text>
      <Text style={styles.durationValue}>{time(row.duration)}</Text>
    </View>)}
  </View>;
}

function dateHeading(timestamp: number) {
  const date = new Date(timestamp);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);
  if (date.toDateString() === today.toDateString()) return 'TODAY';
  if (date.toDateString() === yesterday.toDateString()) return 'YESTERDAY';
  return date.toLocaleDateString(undefined, {month: 'short', day: 'numeric', year: 'numeric'}).toUpperCase();
}

function FormLite() {
  const {width, height} = useWindowDimensions();
  const [insets, setInsets] = useState({top: StatusBar.currentHeight ?? 24, bottom: 24, left: 0, right: 0});
  const [route, go] = useState<Route>('home');
  const [state, setState] = useState(initial);
  const [permission, setPermission] = useState<Permission>('unknown');
  const [history, setHistory] = useState<Summary[]>([]);
  const [selected, select] = useState<Summary | null>(null);
  const [busy, setBusy] = useState(false);
  const [storageError, setStorageError] = useState('');

  useEffect(() => {
    let current = true;
    native.insets().then(value => {
      const measured = JSON.parse(value);
      if (current && measured) setInsets(measured);
    }).catch(() => undefined);
    return () => {current = false;};
  }, [width, height]);

  useEffect(() => {
    const events = new NativeEventEmitter(native);
    const listener = events.addListener('FormLiteState', (value: string) => setState(JSON.parse(value)));
    native.state().then(value => setState(JSON.parse(value))).catch(() => undefined);
    return () => listener.remove();
  }, []);

  useEffect(() => {
    if (route !== 'flow' || permission === 'granted') return;
    const listener = AppState.addEventListener('change', status => {
      if (status === 'active') {
        PermissionsAndroid.check(cameraPermission).then(granted => {
          if (granted) {
            setPermission('granted');
            native.openSetup();
          }
        }).catch(() => undefined);
      }
    });
    return () => listener.remove();
  }, [route, permission]);

  async function cameraSetup() {
    go('flow');
    try {
      const granted = await PermissionsAndroid.check(cameraPermission);
      const result = granted ? PermissionsAndroid.RESULTS.GRANTED : await PermissionsAndroid.request(
        cameraPermission, {
          title: 'Allow camera access',
          message: 'FormLite analyzes the camera on this phone to recognize exercises.',
          buttonPositive: 'Continue', buttonNegative: 'Cancel',
        },
      );
      if (result === PermissionsAndroid.RESULTS.GRANTED) {
        setPermission('granted');
        native.openSetup();
      } else {
        setPermission(result === PermissionsAndroid.RESULTS.NEVER_ASK_AGAIN ? 'blocked' : 'denied');
      }
    } catch {
      setPermission('denied');
    }
  }

  async function showHistory() {
    setStorageError('');
    select(null);
    go('history');
    try {
      setHistory(JSON.parse(await native.sessions()));
    } catch {
      setStorageError('Could not read saved sessions.');
    }
  }

  async function showDetail(id: string) {
    try {
      select(JSON.parse(await native.details(id)));
      go('detail');
    } catch {
      setStorageError('Could not open this session.');
    }
  }

  async function save() {
    if (busy) return;
    setBusy(true);
    setStorageError('');
    try {
      await native.save();
    } catch {
      setStorageError('Could not save your session. Try again.');
    } finally {
      setBusy(false);
    }
  }

  function discard() {
    if (busy) return;
    Alert.alert('Discard this session?', 'The session will be cleared.', [
      {text: 'Keep reviewing', style: 'cancel'},
      {text: 'Discard', style: 'destructive', onPress: () => {
        native.discard();
        setStorageError('');
        go('home');
      }},
    ]);
  }

  function remove(id: string | null) {
    Alert.alert(id ? 'Delete this session?' : 'Delete all history?', 'This removes the saved session data from this phone.', [
      {text: 'Cancel', style: 'cancel'},
      {text: 'Delete', style: 'destructive', onPress: async () => {
        try {
          await native.deleteSession(id);
          await showHistory();
        } catch {
          setStorageError('Could not delete the saved session data.');
        }
      }},
    ]);
  }

  function back() {
    if (route === 'detail') {
      go('history');
    } else if (route === 'flow' && state.canFinish) {
      Alert.alert('End this session?', 'You can review the detected exercise times.', [
        {text: 'Keep going', style: 'cancel'},
        {text: 'End session', onPress: () => native.finish()},
      ]);
    } else if (route === 'flow' && state.phase === 'summary') {
      discard();
    } else {
      if (route === 'flow') native.closeSetup();
      go('home');
    }
  }

  const hardwareBack = useEffectEvent(() => {
    if (route === 'home') return false;
    back();
    return true;
  });

  useEffect(() => {
    const handler = BackHandler.addEventListener('hardwareBackPress', () => hardwareBack());
    return () => handler.remove();
  }, []);

  let content: ReactNode;
  let footer: ReactNode;
  if (route === 'home') {
    content = <>
      <Header title="FormLite" trailing={<Text style={styles.tag}>ON DEVICE</Text>} />
      <View style={styles.introduction}>
        <Text accessibilityRole="header" style={styles.hero}>Know your{'\n'}movement.</Text>
        <Text style={styles.lead}>Four exercises. Your camera.{'\n'}Recognition that stays on your phone.</Text>
      </View>
      <View style={styles.tint}>
        <Text style={styles.tag}>FOUR SUPPORTED EXERCISES</Text>
        <Text style={styles.cardTitle}>Side plank · Squats{'\n'}Reverse crunches · Air jump rope</Text>
        <Text style={styles.secondary}>One person. One camera. No cloud.</Text>
      </View>
      <View style={styles.homeActions}>
        <Button title="Start recognition" press={cameraSetup} />
        <Button title="Exercise guide" outline press={() => go('guide')} />
      </View>
      <Text style={[styles.secondary, styles.homeFootnote]}>No account. No video recording.{'\n'}Works offline after installation.</Text>
    </>;
    footer = <Tabs active="home" home={() => go('home')} history={showHistory} />;
  } else if (route === 'guide') {
    content = <>
      <Header title="Exercise guide" back={back} />
      <Text style={styles.title}>Start with the basics.</Text>
      <Text style={styles.secondary}>Keep your whole body in view.</Text>
      <View style={styles.cards}>{guide.map(([name, guidance]) => <View key={name} style={styles.card}>
        <Text style={styles.historyTitle}>{name}</Text>
        <Text style={styles.secondary}>{guidance}</Text>
      </View>)}</View>
      <Text style={styles.secondary}>Recognition does not assess exercise form.</Text>
    </>;
    footer = <Button title="Set up camera" press={cameraSetup} />;
  } else if (route === 'history') {
    let previous = '';
    content = <>
      <Header title="History" trailing={history.length > 0 && <Pressable accessibilityRole="button" onPress={() => remove(null)}>
        <Text style={styles.tag}>DELETE ALL</Text>
      </Pressable>} />
      <Text style={styles.title}>Your recent sessions</Text>
      <Text style={styles.secondary}>Saved on this phone.</Text>
      {history.length === 0 && !storageError && <View style={styles.cards}>
        <Text style={styles.secondary}>No saved sessions yet.</Text>
        <Button title="Start recognition" press={cameraSetup} />
      </View>}
      {history.map(item => {
        const group = dateHeading(item.started);
        const heading = group !== previous;
        previous = group;
        return <React.Fragment key={item.id}>
          {heading && <Text style={styles.date}>{group}</Text>}
          <Pressable accessibilityRole="button" onPress={() => showDetail(item.id)} style={[styles.card, styles.historyCard]}>
            <Text style={styles.historyTitle}>{time(item.duration)}</Text>
            <Text style={styles.secondary}>{new Date(item.started).toLocaleTimeString(undefined, {hour: 'numeric', minute: '2-digit'})}</Text>
            <Text style={styles.secondary}>{item.exercises.map(entry => entry.name).join(' · ') || 'No exercise recognized'}</Text>
          </Pressable>
        </React.Fragment>;
      })}
      {history.length > 0 && <Text style={[styles.secondary, styles.homeFootnote]}>Tap a session to view its summary.</Text>}
    </>;
    footer = <Tabs active="history" home={() => go('home')} history={showHistory} />;
  } else if ((route === 'detail' && selected) ||
             (route === 'flow' && (state.phase === 'summary' || state.phase === 'saved') && state.draft)) {
    const summary = route === 'detail' ? selected! : state.draft!;
    const saved = route === 'detail' || state.phase === 'saved';
    content = <>
      <Header title={saved ? 'Session saved' : 'Session complete'} back={back} />
      <Text style={saved ? styles.secondary : styles.tag}>{saved ? 'Saved on this phone' : 'NOT SAVED YET'}</Text>
      {!saved && <Text style={styles.summaryTitle}>Your movement,{'\n'}at a glance.</Text>}
      <View style={saved && styles.savedTimer}><TimerCard duration={summary.duration} /></View>
      <DurationTable summary={summary} />
      <Text style={[styles.secondary, styles.summaryNote]}>
        Detected time is an estimate.{'\n'}Repetitions and form are not assessed.
      </Text>
    </>;
    footer = saved ? <>
      <Button title={route === 'detail' ? 'History' : 'New session'} press={route === 'detail' ? showHistory : cameraSetup} />
      <Button title={route === 'detail' ? 'Delete session' : 'History'} outline press={route === 'detail' ? () => remove(summary.id) : showHistory} />
    </> : <>
      <Button title={busy ? 'Saving…' : 'Save session'} press={save} disabled={busy} />
      <Button title="Discard" outline press={discard} disabled={busy} />
    </>;
  } else if (state.phase === 'paused') {
    content = <>
      <Header title="Session paused" />
      <Text style={[styles.hero, styles.pauseTitle]}>Take your time.</Text>
      <Text style={styles.lead}>{state.error ?? 'Camera analysis is paused.\nResume when you are ready.'}</Text>
      <View style={styles.pauseTimer}><TimerCard duration={state.elapsed} /></View>
    </>;
    footer = <>
      <Button title="Resume" press={native.resume} />
      <Button title="Finish" outline press={native.finish} />
    </>;
  } else {
    const active = state.phase === 'active';
    const countdown = state.phase === 'countdown';
    const useCamera = permission === 'granted' && ['setup', 'countdown', 'active'].includes(state.phase);
    content = <>
      <Header title={active ? 'Recognizing' : 'Camera setup'} back={active ? undefined : back} compact={active}
        trailing={active && <Text style={styles.timerTag}>{time(state.elapsed)}</Text>} />
      {!active && <><Text style={styles.title}>Make a little room.</Text>
        <Text style={styles.secondary}>Place your phone on a stable surface.</Text></>}
      <View style={styles.previewSpace}>
        <CameraPane active={active} ready={state.ready && useCamera} enabled={useCamera} countdown={state.countdown} />
      </View>
      {active ? <View style={styles.detection}>
        <Text style={styles.tag}>{state.label ? 'DETECTED EXERCISE' : 'RECOGNITION STATUS'}</Text>
        <Text style={styles.detectTitle} accessibilityLiveRegion="polite">{state.label ?? 'Recognizing…'}</Text>
        <Text style={styles.secondary}>{state.label ? 'Keep moving at your own pace.' : state.message ?? 'Keep your whole body in view.'}</Text>
      </View> : <>
        <Text style={styles.tag}>WHOLE BODY IN VIEW</Text>
        <Text style={[styles.heading, styles.cameraTitle]}>{state.ready ? 'Ready when you are.' : 'Set up your camera.'}</Text>
        <Text accessibilityLiveRegion="polite" style={styles.secondary}>
          {permission === 'blocked' || permission === 'denied'
            ? 'Camera access is needed to recognize exercises.'
            : state.error ?? 'Video is analyzed here, never recorded.'}
        </Text>
      </>}
      <View style={styles.cameraActions}>
        {active ? <View style={styles.row}>
          <Button title="Pause" outline press={native.pause} />
          <Button title="Finish" press={native.finish} />
        </View> : <>
          {permission === 'blocked' ? <Button title="Open settings" press={() => Linking.openSettings()} /> :
            permission === 'denied' ? <Button title="Retry camera access" press={cameraSetup} /> :
            <>
              <Button title={state.error ? 'Retry camera' : 'Switch camera'} outline
                press={state.error ? native.openSetup : native.switchCamera} disabled={countdown || !useCamera} />
              <Button title="Start session" press={native.start} disabled={!state.ready || countdown} />
            </>}
        </>}
      </View>
      {active && <Text style={[styles.secondary, styles.liveNote]}>Exercise recognition only · No rep counting</Text>}
    </>;
  }

  return <View style={[styles.safe, {paddingTop: insets.top, paddingBottom: insets.bottom, paddingLeft: insets.left, paddingRight: insets.right}]}>
    <StatusBar barStyle="dark-content" />
    <ScrollView contentContainerStyle={styles.page} keyboardShouldPersistTaps="handled">
      {content}
      {!!storageError && (route === 'history' || route === 'detail' || state.phase === 'summary') &&
        <Text style={styles.error} accessibilityLiveRegion="polite">{storageError}</Text>}
      {footer && <View style={styles.footer}>{footer}</View>}
    </ScrollView>
  </View>;
}

export default function App() {
  return <FormLite />;
}

const styles = StyleSheet.create({
  safe: {flex: 1, backgroundColor: colors.bg},
  page: {flexGrow: 1, paddingHorizontal: 24, paddingTop: 26, paddingBottom: 14},
  row: {flexDirection: 'row', gap: 14, alignItems: 'center'},
  header: {minHeight: 32, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 30},
  liveHeader: {marginBottom: 17},
  headingPress: {minHeight: 40, justifyContent: 'center'},
  heading: {fontSize: 24, fontWeight: '700', lineHeight: 30, color: colors.text},
  tag: {color: colors.green, fontSize: 11, lineHeight: 15, fontWeight: '700'},
  hero: {fontSize: 34, lineHeight: 42, fontWeight: '700', color: colors.text},
  title: {fontSize: 26, lineHeight: 34, fontWeight: '700', color: colors.text},
  lead: {fontSize: 16, lineHeight: 20, color: colors.muted, marginTop: 13},
  secondary: {fontSize: 14, lineHeight: 18, color: colors.muted},
  introduction: {marginBottom: 35},
  tint: {backgroundColor: colors.pale, borderRadius: 20, padding: 20, gap: 6},
  cardTitle: {fontSize: 23, lineHeight: 29, fontWeight: '700', color: colors.text},
  button: {minHeight: 54, flexGrow: 1, flexBasis: 0, borderRadius: 15, paddingHorizontal: 14,
    borderWidth: 1, borderColor: colors.green, backgroundColor: colors.green, justifyContent: 'center', alignItems: 'center'},
  buttonText: {fontSize: 16, fontWeight: '500', color: colors.white, textAlign: 'center', paddingVertical: 10},
  outline: {backgroundColor: colors.white, borderColor: colors.border},
  outlineText: {color: colors.green},
  dim: {opacity: .54},
  homeActions: {marginTop: 42, gap: 12},
  homeFootnote: {marginTop: 34},
  footer: {marginTop: 'auto', paddingTop: 30, gap: 12},
  tabs: {marginTop: 0},
  cards: {gap: 6, marginTop: 14, marginBottom: 24},
  card: {borderWidth: 1, borderColor: colors.border, borderRadius: 20, backgroundColor: colors.white,
    paddingHorizontal: 20, paddingVertical: 16, minHeight: 105, gap: 6},
  historyTitle: {fontSize: 18, lineHeight: 24, fontWeight: '700', color: colors.text},
  date: {fontSize: 11, fontWeight: '700', color: colors.muted, marginTop: 24, marginBottom: 13},
  historyCard: {marginBottom: 18},
  previewSpace: {marginTop: 12, marginBottom: 18},
  camera: {backgroundColor: colors.camera, borderRadius: 22, overflow: 'hidden'},
  preview: {position: 'absolute', top: 0, left: 0, width: '100%', height: '100%'},
  cameraCover: {backgroundColor: colors.camera, alignItems: 'center'},
  illustration: {width: 342, height: 340, position: 'absolute'},
  previewCaption: {position: 'absolute', bottom: 14, color: colors.white, fontSize: 11},
  cameraTitle: {marginTop: 15, marginBottom: 2},
  cameraActions: {marginTop: 16, gap: 12},
  timerTag: {color: colors.green, fontSize: 14, fontWeight: '700', fontVariant: ['tabular-nums']},
  countdown: {...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center', backgroundColor: '#172c2980'},
  countdownText: {fontSize: 74, color: colors.white, fontWeight: '700'},
  detection: {borderRadius: 22, padding: 20, gap: 8, backgroundColor: colors.white, minHeight: 153},
  detectTitle: {color: colors.text, fontSize: 35, lineHeight: 43, fontWeight: '700'},
  liveNote: {marginTop: 40, fontSize: 12},
  pauseTitle: {marginTop: 26},
  pauseTimer: {marginTop: 40},
  summaryTitle: {fontSize: 30, lineHeight: 38, color: colors.text, fontWeight: '700', marginTop: 26, marginBottom: 12},
  savedTimer: {marginTop: 26},
  largeTime: {fontSize: 42, fontWeight: '700', color: colors.text, fontVariant: ['tabular-nums']},
  durationTable: {marginTop: 18},
  durationRow: {minHeight: 55, justifyContent: 'space-between', borderBottomWidth: StyleSheet.hairlineWidth, borderColor: colors.border},
  durationLabel: {color: colors.text, fontSize: 16, fontWeight: '500', flex: 1},
  durationValue: {color: colors.text, fontSize: 16, fontWeight: '700', fontVariant: ['tabular-nums']},
  summaryNote: {marginTop: 30},
  error: {marginTop: 12, fontSize: 14, color: '#a12d24'},
});
