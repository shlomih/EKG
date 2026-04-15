/**
 * DisclaimerBanner — reusable clinical disclaimer
 *
 * CRITICAL: Must appear on every analysis result screen (HIPAA 5.3.5, FDA 7.1)
 * Fixed text, consistent styling across app
 */

import { View, Text, StyleSheet } from 'react-native';
import { useTranslation } from 'react-i18next';

export default function DisclaimerBanner() {
  const { t } = useTranslation();

  return (
    <View style={styles.container}>
      <View style={styles.warningDot} />
      <Text style={styles.text}>{t('disclaimer')}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: '#1a1a2e',
    borderRadius: 8,
    padding: 12,
    marginVertical: 16,
    borderLeftWidth: 3,
    borderLeftColor: '#FF6B6B',
    alignItems: 'flex-start',
  },
  warningDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#FF6B6B',
    marginRight: 8,
    marginTop: 4,
  },
  text: {
    flex: 1,
    fontSize: 11,
    color: '#8B8B8B',
    lineHeight: 16,
  },
});
