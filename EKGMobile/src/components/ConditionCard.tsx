/**
 * ConditionCard — display detected condition with urgency indicator
 *
 * Layout:
 * - Left: colored urgency bar (width=4px, height=full)
 * - Main content: condition name, probability %, action text
 * - Adaptable to dark theme
 */

import { View, Text, StyleSheet } from 'react-native';
import { DetectedCondition } from '@/src/types/Analysis';

interface ConditionCardProps {
  condition: DetectedCondition;
  onPress?: () => void;
}

export default function ConditionCard({ condition }: ConditionCardProps) {
  return (
    <View
      style={[
        styles.container,
        { borderLeftColor: condition.urgencyColor },
      ]}
    >
      {/* Urgency color bar (left side) */}
      <View
        style={[
          styles.urgencyBar,
          { backgroundColor: condition.urgencyColor },
        ]}
      />

      {/* Content area */}
      <View style={styles.content}>
        {/* Header: name + probability */}
        <View style={styles.header}>
          <Text style={styles.conditionName}>{condition.name}</Text>
          <View style={[styles.probabilityBadge, { backgroundColor: condition.urgencyColor + '20' }]}>
            <Text style={[styles.probabilityText, { color: condition.urgencyColor }]}>
              {(condition.probability * 100).toFixed(0)}%
            </Text>
          </View>
        </View>

        {/* Urgency label */}
        <Text style={[styles.urgencyLabel, { color: condition.urgencyColor }]}>
          {condition.urgencyLabel}
        </Text>

        {/* Action text */}
        {condition.action && (
          <Text style={styles.actionText}>{condition.action}</Text>
        )}

        {/* Clinical note */}
        {condition.note && (
          <Text style={styles.noteText}>{condition.note}</Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: '#0D1F1E',
    borderRadius: 8,
    overflow: 'hidden',
    marginBottom: 10,
    borderLeftWidth: 4,
  },
  urgencyBar: {
    width: 4,
  },
  content: {
    flex: 1,
    padding: 12,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  conditionName: {
    fontSize: 15,
    fontWeight: '600',
    color: '#E0E0E0',
    flex: 1,
  },
  probabilityBadge: {
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
    marginLeft: 8,
  },
  probabilityText: {
    fontSize: 12,
    fontWeight: '600',
  },
  urgencyLabel: {
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 4,
  },
  actionText: {
    fontSize: 12,
    color: '#5A8A85',
    marginBottom: 2,
    lineHeight: 16,
  },
  noteText: {
    fontSize: 11,
    color: '#3D6662',
    fontStyle: 'italic',
    marginTop: 4,
    lineHeight: 14,
  },
});
