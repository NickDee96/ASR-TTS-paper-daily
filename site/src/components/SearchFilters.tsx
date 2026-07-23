import {
  Button,
  DrawerBody,
  DrawerHeader,
  DrawerHeaderTitle,
  Field,
  OverlayDrawer,
  Select,
} from '@fluentui/react-components';
import { Filter, X } from 'lucide-react';
import { useRef } from 'react';
import type { SearchUrlState } from '../lib/search-state';
import { activeFilterCount } from '../lib/search-state';

type FacetCounts = Record<string, Record<string, number>>;

interface SearchFiltersProps {
  state: SearchUrlState;
  facets: FacetCounts;
  drawerOpen: boolean;
  onDrawerOpenChange(open: boolean): void;
  onChange(patch: Partial<SearchUrlState>): void;
  onClear(): void;
}

interface FilterFieldsProps {
  state: SearchUrlState;
  facets: FacetCounts;
  idPrefix: string;
  onChange(patch: Partial<SearchUrlState>): void;
}

interface ActiveFilter {
  key: keyof SearchUrlState | 'years';
  label: string;
}

function valuesFor(facets: FacetCounts, name: string): Array<[string, number]> {
  return Object.entries(facets[name] ?? {})
    .sort(([left], [right]) => left.localeCompare(right));
}

function FilterFields({ state, facets, idPrefix, onChange }: FilterFieldsProps) {
  const categories = valuesFor(facets, 'category');
  const years = valuesFor(facets, 'year')
    .filter(([year]) => /^\d{4}$/.test(year))
    .sort(([left], [right]) => right.localeCompare(left));
  const codeValues = valuesFor(facets, 'code');
  const statusValues = valuesFor(facets, 'status');
  const recordValues = valuesFor(facets, 'record_status');
  return (
    <div className="filter-fields">
      <Field label="Category">
        <Select
          id={`${idPrefix}-category`}
          value={state.category}
          disabled={categories.length === 0}
          onChange={(event) => onChange({ category: event.target.value })}
        >
          <option value="">{categories.length ? 'All categories' : 'Awaiting metadata'}</option>
          {categories.map(([value, count]) => <option key={value} value={value}>{value} ({count})</option>)}
        </Select>
      </Field>
      <Field label="Code">
        <Select id={`${idPrefix}-code`} value={state.code} onChange={(event) => onChange({ code: event.target.value })}>
          <option value="">Any code status</option>
          {codeValues.map(([value, count]) => <option key={value} value={value}>{value} ({count})</option>)}
        </Select>
      </Field>
      <Field label="Paper status">
        <Select id={`${idPrefix}-status`} value={state.status} onChange={(event) => onChange({ status: event.target.value })}>
          <option value="">New and revised</option>
          {statusValues.map(([value, count]) => <option key={value} value={value}>{value} ({count})</option>)}
        </Select>
      </Field>
      <Field label="Metadata">
        <Select id={`${idPrefix}-metadata`} value={state.recordStatus} onChange={(event) => onChange({ recordStatus: event.target.value })}>
          <option value="">Any completeness</option>
          {recordValues.map(([value, count]) => <option key={value} value={value}>{value} ({count})</option>)}
        </Select>
      </Field>
      <Field label="From year">
        <Select id={`${idPrefix}-from`} value={state.fromYear} onChange={(event) => onChange({ fromYear: event.target.value })}>
          <option value="">Any year</option>
          {years.map(([value, count]) => <option key={value} value={value}>{value} ({count})</option>)}
        </Select>
      </Field>
      <Field label="To year">
        <Select id={`${idPrefix}-to`} value={state.toYear} onChange={(event) => onChange({ toYear: event.target.value })}>
          <option value="">Any year</option>
          {years.map(([value, count]) => <option key={value} value={value}>{value} ({count})</option>)}
        </Select>
      </Field>
    </div>
  );
}

export default function SearchFilters({
  state,
  facets,
  drawerOpen,
  onDrawerOpenChange,
  onChange,
  onClear,
}: SearchFiltersProps) {
  const triggerReference = useRef<HTMLButtonElement>(null);
  const filterCount = activeFilterCount(state);
  const activeFilters: ActiveFilter[] = [
    state.topic && { key: 'topic', label: `Topic: ${state.topic}` },
    state.category && { key: 'category', label: `Category: ${state.category}` },
    state.code && { key: 'code', label: `Code: ${state.code}` },
    state.status && { key: 'status', label: `Status: ${state.status}` },
    state.recordStatus && { key: 'recordStatus', label: `Metadata: ${state.recordStatus}` },
    (state.fromYear || state.toYear) && {
      key: 'years',
      label: `Years: ${state.fromYear || 'earliest'}-${state.toYear || 'latest'}`,
    },
  ].filter((filter): filter is ActiveFilter => Boolean(filter));

  const removeFilter = (filter: ActiveFilter) => {
    if (filter.key === 'years') {
      onChange({ fromYear: '', toYear: '' });
      return;
    }
    onChange({ [filter.key]: '' });
  };
  return (
    <>
      <div className="filter-command-bar">
        <Button
          ref={triggerReference}
          className="mobile-filter-trigger"
          icon={<Filter aria-hidden="true" size={17} />}
          onClick={() => onDrawerOpenChange(true)}
        >
          Filters{filterCount ? ` (${filterCount})` : ''}
        </Button>
        <Field className="sort-field" label="Sort" orientation="horizontal">
          <Select
            aria-label="Sort results"
            value={state.sort}
            onChange={(event) => onChange({ sort: event.target.value as SearchUrlState['sort'] })}
          >
            <option value="relevance">Relevance</option>
            <option value="newest">Newest published</option>
            <option value="updated">Recently updated</option>
          </Select>
        </Field>
        {filterCount > 0 && <Button appearance="subtle" onClick={onClear}>Clear filters</Button>}
      </div>
      {activeFilters.length > 0 && (
        <div className="active-filters" aria-label="Active filters">
          {activeFilters.map((filter) => (
            <Button
              key={filter.key}
              appearance="subtle"
              size="small"
              icon={<X aria-hidden="true" size={14} />}
              iconPosition="after"
              aria-label={`Remove ${filter.label} filter`}
              onClick={() => removeFilter(filter)}
            >
              {filter.label}
            </Button>
          ))}
        </div>
      )}
      <div className="desktop-filter-panel" aria-label="Search filters">
        <FilterFields state={state} facets={facets} idPrefix="desktop" onChange={onChange} />
      </div>
      <OverlayDrawer
        position="end"
        open={drawerOpen}
        onOpenChange={(_, data) => {
          onDrawerOpenChange(data.open);
          if (!data.open) {
            window.requestAnimationFrame(() => triggerReference.current?.focus());
          }
        }}
      >
        <DrawerHeader>
          <DrawerHeaderTitle
            action={(
              <Button
                appearance="subtle"
                icon={<X aria-hidden="true" size={18} />}
                aria-label="Close filters"
                onClick={() => onDrawerOpenChange(false)}
              />
            )}
          >
            Filter papers
          </DrawerHeaderTitle>
        </DrawerHeader>
        <DrawerBody>
          <FilterFields state={state} facets={facets} idPrefix="mobile" onChange={onChange} />
          <div className="drawer-actions">
            <Button appearance="primary" onClick={() => onDrawerOpenChange(false)}>Show results</Button>
            {filterCount > 0 && <Button appearance="subtle" onClick={onClear}>Clear all</Button>}
          </div>
        </DrawerBody>
      </OverlayDrawer>
    </>
  );
}