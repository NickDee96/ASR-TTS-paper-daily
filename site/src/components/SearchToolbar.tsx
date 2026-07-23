import { Button, FluentProvider, Input, Tooltip, webLightTheme } from '@fluentui/react-components';
import { ArrowRight, Code, Search } from 'lucide-react';

interface SearchToolbarProps {
  action: string;
  initialQuery?: string;
}

const researchTheme = {
  ...webLightTheme,
  colorBrandBackground: '#176b5f',
  colorBrandBackgroundHover: '#10584f',
  colorBrandBackgroundPressed: '#0d4942',
  colorBrandForeground1: '#176b5f',
  colorCompoundBrandForeground1: '#176b5f',
  fontFamilyBase: '"Instrument Sans", sans-serif',
  borderRadiusMedium: '4px',
  borderRadiusLarge: '6px',
};

export default function SearchToolbar({ action, initialQuery = '' }: SearchToolbarProps) {
  const query = new URLSearchParams(window.location.search).get('q') ?? initialQuery;
  return (
    <FluentProvider theme={researchTheme} className="fluent-toolbar-provider">
      <form className="search-toolbar" action={action} method="get" role="search">
        <Input
          className="search-input"
          name="q"
          defaultValue={query}
          contentBefore={<Search aria-hidden="true" size={18} />}
          placeholder="Search titles, abstracts, authors, or arXiv IDs"
          aria-label="Search the paper archive"
          size="large"
        />
        <Button appearance="primary" type="submit" icon={<ArrowRight aria-hidden="true" size={18} />}>
          Search
        </Button>
        <Tooltip content="Open the source repository" relationship="label">
          <Button
            as="a"
            href="https://github.com/NickDee96/ASR-TTS-paper-daily"
            target="_blank"
            rel="noreferrer"
            appearance="subtle"
            icon={<Code aria-hidden="true" size={18} />}
          >
            Repository
          </Button>
        </Tooltip>
      </form>
    </FluentProvider>
  );
}